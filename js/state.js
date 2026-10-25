// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · el estado que sobrevive entre tokens.
// ─────────────────────────────────────────────────────────────────────────────
// En qwen35 son DOS cosas distintas, y esa asimetría es la que hace barato el
// contexto largo:
//
//   · 6 capas de atención → caché KV, que CRECE con el contexto.
//   · 18 capas SSM → estado recurrente de tamaño FIJO (128×128 por cabeza) más
//     una ventana de convolución de 3 tokens. No crece nunca.
//
// Cuentas del modelo de 0.8B, sacadas de sus propios metadatos:
//   · las 6 capas de atención guardan K y V de cada token: la caché KV crece en
//     línea recta con el contexto;
//   · las 18 capas SSM guardan una matriz de 128×128 por cabeza: fija;
//   · la ventana de convolución es otra parte fija, mucho más pequeña.
//
// O sea: el estado recurrente de las 18 capas cuesta menos que 800 tokens de
// caché KV. Una cuenta de KV que supusiera las 24 capas de atención sobraría
// por dos órdenes de magnitud.
//
// PENDIENTE de verdad (no lo disimulo): la caché se reserva entera al abrir la
// sesión, con la longitud de contexto que se pida. Para los 262.144 de contexto
// que declara el modelo harían falta páginas: la caché de todo ese contexto no
// se reserva de una sentada. Con contextos de conversación normales esto sirve
// tal cual.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Del modulo de shaders, que es texto puro: importar ops.js aqui arrastraria
// las globales de WebGPU y state.js dejaria de poder inspeccionarse desde Node.
import { SSM_HEAD_DIM } from './kernels.ssm.wgsl.js';

const MiB = 1024 ** 2;

export function createState(device, hp, opts = {}) {
  const ctxLen = opts.contextLen || 2048;
  if (ctxLen > hp.contextLength) {
    throw new Error(`contexto ${ctxLen} mayor que el del modelo (${hp.contextLength})`);
  }
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buf = (bytes, label) => device.createBuffer({ label, size: bytes, usage });

  // Se decide AQUÍ y se expone en el estado, para que el shader que escribe la
  // caché y los que la leen no puedan discrepar. Si uno escribiera f16 y otro
  // leyera f32, no habría error: habría basura.
  const kvF16 = opts.kvF16 !== undefined ? !!opts.kvF16
    : globalThis.__ELFFUSS_KV_F16 !== false;
  const bytesKv = kvF16 ? 2 : 4;

  const kv = new Map();      // índice de capa → {k, v}
  const ssm = new Map();     // índice de capa → {state, conv}
  let bytes = 0;

  for (const l of hp.layers) {
    if (l.kind === 'attn') {
      // Media precisión en la caché KV: la mitad de memoria y la mitad de
      // tráfico al leerla. Es el escalón que va ANTES de cualquier cuantización
      // —f32 no es siquiera la línea base de la industria, que es f16— y el
      // único que no plantea preguntas de precisión más allá del redondeo.
      const per = ctxLen * hp.kvHeads * hp.headDim * bytesKv;
      kv.set(l.index, { k: buf(per, `k${l.index}`), v: buf(per, `v${l.index}`) });
      bytes += per * 2;
    } else {
      // Un estado de S_k x S_v por cabeza de VALOR, no de clave.
      const st = hp.ssmVHeads * hp.ssmState * hp.ssmState * 4;
      const cv = (hp.convKernel - 1) * hp.convChannels * 4;
      ssm.set(l.index, { state: buf(st, `s${l.index}`), conv: buf(cv, `c${l.index}`) });
      bytes += st + cv;
    }
  }

  // Buffers sombra para las instantáneas del estado recurrente. Se crean UNA vez
  // por ranura, la primera que se pida, y se reutilizan: crear ~96 buffers
  // dentro de un turno es de las cosas que se llevan la pestaña por delante.
  // `past` a -1 significa «esta ranura no tiene instantánea válida».
  //
  // Hay RANURAS CON NOMBRE y no una sola, y la razón es la mitad del asunto. La
  // reutilización de prefijo exige que el prompt anterior COMPLETO sea prefijo
  // estricto del nuevo, y con una sola foto —tomada al final del prompt
  // anterior— eso solo se cumple si la conversación crece por el final y nada
  // de en medio cambia. En un arnés siempre cambia algo en medio (el estado
  // vivo), así que la foto única no servía nunca: medido x1,00.
  //
  // Con dos ranuras hay dos niveles de rescate:
  //   · 'turno'   la de siempre: el final del prompt anterior. Caso ideal.
  //   · 'sistema' tomada justo después del prompt de sistema, y SOBREVIVE a
  //               `reset()`. Cuando el prefijo completo falla, se vuelve aquí y
  //               se reprocesa solo la conversación, no las ~1.300-2.200 fichas
  //               del prompt de sistema.
  //
  // Coste: cada ranura es una copia del estado recurrente, FIJA: no crece con
  // el contexto. En el modelo grande es unas ocho veces la del pequeño, y aun
  // así dos ranuras no llegan al 5 % de lo que pesan sus pesos; se cambian por
  // no reprocesar el prompt de sistema en cada turno.
  const ranuras = new Map();
  const ranura = (nombre) => {
    let r = ranuras.get(nombre);
    if (!r) { r = { sombra: null, past: -1 }; ranuras.set(nombre, r); }
    return r;
  };

  const state = {
    ctxLen,
    nPast: 0,
    bytes,
    kv, ssm,
    // Lo lee el forward para pedir el shader que corresponde. No es decorativo:
    // escribir y leer la caché con precisiones distintas no da error, da basura.
    kvF16,

    // Arrancar una conversación nueva: el estado recurrente vuelve a cero y la
    // caché KV se olvida. Es OBLIGATORIO entre secuencias — un estado SSM
    // arrastrado de otra conversación no da error, da respuestas contaminadas.
    reset() {
      state.nPast = 0;
      for (const { state: s, conv } of ssm.values()) {
        device.queue.writeBuffer(s, 0, new Float32Array(s.size / 4));
        device.queue.writeBuffer(conv, 0, new Float32Array(conv.size / 4));
      }
      // La KV no hace falta limpiarla: la máscara causal solo mira hasta nPast.
      //
      // La instantánea del TURNO deja de valer: describe un estado que ya no
      // existe, y restaurarla después de esto devolvería la conversación
      // ANTERIOR sin dar ningún error. Se invalida aquí y no en quien llama,
      // que es donde se olvidaría.
      ranura('turno').past = -1;
      // ⚠️ La del SYSTEM **no** se invalida, y es a propósito: describe «el
      // estado después del prompt de sistema», que no depende de la
      // conversación. Es justo lo que la hace útil — sobrevive al reset y
      // permite volver ahí en vez de reprocesar el prompt entero. Quien la
      // invalida es quien cambia el prompt de sistema, comparando fichas
      // (provider.js), no esta función.
    },

    advance(n) {
      if (state.nPast + n > ctxLen) {
        throw new Error(`la caché se llena: ${state.nPast + n} tokens sobre ${ctxLen} de contexto`);
      }
      state.nPast += n;
    },

    // ── Fotografiar el estado para poder volver aquí ──────────────────────────
    // Sirve para no reprocesar la conversación entera en cada turno. Lo que se
    // copia es SOLO el estado recurrente del SSM, y eso es deliberado:
    //
    //   · El recurrente NO se rebobina: es el resultado de haber recorrido toda
    //     la secuencia, así que la única forma de volver atrás es tener copia.
    //   · La KV NO se copia. La máscara causal descarta con un CERO EXACTO todo
    //     k > qAbs = nPast + fila — está así en `kernels.nn.wgsl.js` y en
    //     `softmaxCPU`, escritas por separado y coincidentes—, de modo que
    //     restaurar una posición anterior es devolver `nPast` atrás y dejar que
    //     lo viejo se sobrescriba. Es lo mismo en lo que ya se apoya `reset()`.
    //
    // La diferencia no es cosmética: copiar la KV costaría mucho más que el
    // estado recurrente, y además CRECE con el contexto, mientras el estado
    // recurrente es de tamaño FIJO. Fotografiar una conversación de 200.000
    // tokens cuesta exactamente lo mismo que una de 10.
    instantanea(nombre = 'turno') {
      const r = ranura(nombre);
      if (!r.sombra) {
        r.sombra = new Map();
        for (const [i, { state: s, conv }] of ssm) {
          r.sombra.set(i, { state: buf(s.size, `snap-${nombre}-s${i}`), conv: buf(conv.size, `snap-${nombre}-c${i}`) });
        }
      }
      const enc = device.createCommandEncoder({ label: `instantanea:${nombre}` });
      for (const [i, { state: s, conv }] of ssm) {
        const d = r.sombra.get(i);
        enc.copyBufferToBuffer(s, 0, d.state, 0, s.size);
        enc.copyBufferToBuffer(conv, 0, d.conv, 0, conv.size);
      }
      device.queue.submit([enc.finish()]);
      r.past = state.nPast;
      return r.past;
    },

    // Devuelve false si no hay instantánea válida, para que quien llame reinicie
    // en vez de seguir con un estado que no sabe describir.
    restaura(nombre = 'turno') {
      const r = ranuras.get(nombre);
      if (!r || !r.sombra || r.past < 0) return false;
      const enc = device.createCommandEncoder({ label: `restaura:${nombre}` });
      for (const [i, { state: s, conv }] of ssm) {
        const d = r.sombra.get(i);
        enc.copyBufferToBuffer(d.state, 0, s, 0, s.size);
        enc.copyBufferToBuffer(d.conv, 0, conv, 0, conv.size);
      }
      device.queue.submit([enc.finish()]);
      state.nPast = r.past;
      return true;
    },

    tieneInstantanea(nombre = 'turno') {
      const r = ranuras.get(nombre);
      return !!r && r.past >= 0;
    },

    // Cuántas fichas describe una ranura. Hace falta fuera: quien reutiliza
    // tiene que saber desde qué posición sigue, y adivinarlo sería volver al
    // fallo de tener la cuenta en dos sitios.
    pasadoInstantanea(nombre = 'turno') {
      const r = ranuras.get(nombre);
      return r ? r.past : -1;
    },

    olvidaInstantanea(nombre) {
      if (nombre === undefined) { for (const r of ranuras.values()) r.past = -1; return; }
      const r = ranuras.get(nombre);
      if (r) r.past = -1;
    },

    summary() {
      const kvB = [...kv.values()].reduce((n, x) => n + x.k.size + x.v.size, 0);
      const ssmB = [...ssm.values()].reduce((n, x) => n + x.state.size + x.conv.size, 0);
      return `${kv.size} capas KV (${(kvB / MiB).toFixed(0)} MiB a ${ctxLen} de contexto, ${kvF16 ? 'f16' : 'f32'}) · ` +
        `${ssm.size} capas SSM (${(ssmB / MiB).toFixed(1)} MiB fijos)`;
    },

    destroy() {
      for (const x of kv.values()) { x.k.destroy(); x.v.destroy(); }
      for (const x of ssm.values()) { x.state.destroy(); x.conv.destroy(); }
      // Y los buffers sombra de la instantánea, que se crean aparte y aquí no
      // estaban: en el 27B son una copia entera del estado recurrente, y se
      // filtrarían en CADA ciclo de descarga y recarga. El detalle que lo hace
      // peor de lo que parece es QUIÉN llama a esto: la app invoca `unload()` justo cuando
      // la memoria va justa, así que la fuga aparecería en el peor momento
      // posible y disfrazada de «el navegador mató la pestaña».
      // TODAS las ranuras, no una: desde que hay dos, liberar solo la primera
      // dejaría la otra colgada, y es el mismo tamaño.
      for (const r of ranuras.values()) {
        if (!r.sombra) continue;
        for (const x of r.sombra.values()) { x.state.destroy(); x.conv.destroy(); }
        r.sombra = null;
      }
      ranuras.clear();
    },
  };
  state.reset();
  return state;
}

// Lee los hiperparámetros del GGUF y clasifica los bloques. Todo sale del
// fichero: nada cableado, porque el 27B tiene 65 bloques y el 0.8B 24, y el
// patrón de atención sale de su propia clave.
// Arquitecturas que este motor sabe ejecutar, y QUÉ las diferencia.
//
// `qwen35` es híbrida: la mayoría de bloques son recurrentes y uno de cada
// cuatro es de atención. `llama` es atención pura — y resulta ser un
// SUBCONJUNTO, no un camino aparte: el bloque de atención de qwen35 ya hace
// casi exactamente lo que pide llama.
//
// Las diferencias se declaran como DATOS y no como ramas sueltas por el
// forward, porque cada rama nueva en el camino de qwen35 es una oportunidad de
// romper los dos modelos que hoy están en producción:
//
//   qGate     qwen35 proyecta `attn_q` al DOBLE (Q y una puerta intercalados
//             por cabeza) y multiplica la salida por esa puerta. llama no.
//   qkNorm    qwen35 normaliza Q y K antes de RoPE. llama no trae esos tensores.
//   ropeMode  ⚠️ EL PELIGROSO. qwen35 empareja como NEOX; `llama` usa NORM
//             (pares adyacentes) porque el conversor a GGUF ya permuta los
//             pesos. Equivocarse aquí NO peta: da texto gramatical que se
//             degrada al crecer el contexto, que es la peor forma de fallar.
//   ffnNorm   se llama `post_attention_norm` en qwen35 y `ffn_norm` en llama.
//
// Los nombres NO se deducen mirando si el tensor existe. Con `has(...)`, un
// modelo al que le faltara el tensor dejaría de fallar y empezaría a mentir.
const ARQUITECTURAS = {
  qwen35: { qGate: true,  qkNorm: true,  ropeMode: 'neox', ffnNorm: 'post_attention_norm', hibrida: true },
  llama:  { qGate: false, qkNorm: false, ropeMode: 'norm', ffnNorm: 'ffn_norm',            hibrida: false },
};

export function readHParams(meta, tensors = null) {
  const arch = meta['general.architecture'];
  const perfil = ARQUITECTURAS[arch];
  if (!perfil) {
    throw new Error(`arquitectura no soportada: ${arch} (conocidas: ${Object.keys(ARQUITECTURAS).join(', ')})`);
  }
  const g = (k, def) => {
    const v = meta[`${arch}.${k}`];
    if (v === undefined && def === undefined) throw new Error(`falta la clave qwen35.${k}`);
    return v === undefined ? def : v;
  };

  const nLayersFichero = g('block_count');
  // PREDICCIÓN MULTI-TOKEN. El 27B declara nextn_predict_layers=1: el ÚLTIMO
  // bloque no pertenece al forward, es una cabeza que predice el token n+2. Si
  // se ejecuta como una capa normal, el modelo genera tokens reales pero
  // degenerados («if if if if…») — no peta, solo deja de tener sentido.
  //
  // Y explica la anomalía que había atribuido a otra cosa: con 65 bloques, la
  // regla «uno de cada cuatro es atención» daba 16 y el fichero tenía 17. No
  // había excepción: el 17º era la capa MTP, que no cuenta.
  const nextn = meta['qwen35.nextn_predict_layers'] || 0;
  const nLayers = nLayersFichero - nextn;
  const interval = g('full_attention_interval', 4);
  const heads = g('attention.head_count');
  const kvHeads = g('attention.head_count_kv');
  const embd = g('embedding_length');
  // head_dim NO es embd/heads en este modelo: 1024/8 = 128, pero el head_dim
  // real es 256. Sale de su propia clave.
  const headDim = g('attention.key_length', Math.floor(embd / heads));

  // Se clasifica por los tensores que el fichero TIENE, no por una fórmula.
  // La fórmula es respaldo y coincide exactamente con la de la referencia:
  // atención cuando (i+1) % interval == 0, recurrente en el resto. El bloque MTP
  // también trae attn_q, así que entra en este conjunto — da igual, porque el
  // bucle solo pregunta por los `nLayers` que sí se ejecutan.
  const attnSet = tensors
    ? new Set(tensors
      .map(t => /^blk\.(\d+)\.attn_q\.weight$/.exec(t.name))
      .filter(Boolean).map(m => +m[1]))
    : null;
  const esAtencion = (i) => attnSet
    ? attnSet.has(i)
    : (i + 1) % interval === 0;

  const layers = [];
  for (let i = 0; i < nLayers; i++) {
    layers.push({ index: i, kind: esAtencion(i) ? 'attn' : 'ssm' });
  }

  // DOS recuentos distintos, y confundirlos rompe solo los modelos grandes:
  // `group_count` son las cabezas de CLAVE (16 en los dos modelos) y
  // `time_step_rank` las de VALOR (16 en el 0.8B, 48 en el 27B). En el pequeño
  // coinciden, así que usar uno para todo funciona ahí y solo ahí.
  const ssmHeads = g('ssm.group_count', 16);
  const ssmVHeads = g('ssm.time_step_rank', ssmHeads);
  const ssmState = g('ssm.state_size', SSM_HEAD_DIM);
  const ssmInner = g('ssm.inner_size', 2048);

  // Dos supuestos que el motor entero da por ciertos y que aquí se comprueban en
  // vez de descubrirse como texto raro: que las cabezas de valor son múltiplo de
  // las de clave (si no, el reparto q/k no está definido) y que la dimensión de
  // value coincide con la de key (el estado es cuadrado y el shader usa una sola
  // S para las dos). Los dos se cumplen en el 0.8B y en el 27B.
  if (ssmVHeads % ssmHeads !== 0) {
    throw new Error(`cabezas SSM incompatibles: ${ssmVHeads} de valor no es ` +
      `múltiplo de ${ssmHeads} de clave`);
  }
  const headV = ssmInner / ssmVHeads;
  if (headV !== ssmState) {
    throw new Error(`el estado SSM no es cuadrado: value_dim ${headV} ` +
      `(inner ${ssmInner} / ${ssmVHeads} cabezas) contra key_dim ${ssmState}`);
  }

  return {
    arch, ...perfil, nLayers, interval, layers, nextn, nLayersFichero,
    embd, heads, kvHeads, headDim,
    ffnLength: g('feed_forward_length'),
    contextLength: g('context_length'),
    ropeFreqBase: g('rope.freq_base', 1e7),
    ropeDims: g('rope.dimension_count', 64),
    eps: meta[`${arch}.attention.layer_norm_rms_epsilon`] ?? 1e-6,
    ssmHeads, ssmVHeads, ssmState, ssmInner,
    convKernel: g('ssm.conv_kernel', 4),
    // conv_dim = 2·key_dim + value_dim, con key_dim = grupos·estado y
    // value_dim = inner. En el 0.8B los dos valen 2048 y da 6144; en el 27B el
    // value_dim son 6144 y da 10240.
    convChannels: 2 * ssmHeads * ssmState + ssmInner,
    nAttn: layers.filter(l => l.kind === 'attn').length,
    nSSM: layers.filter(l => l.kind === 'ssm').length,
  };
}
