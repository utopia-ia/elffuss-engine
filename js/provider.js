// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · el adaptador que enchufa el motor al agente de Elffuss.
// ─────────────────────────────────────────────────────────────────────────────
// El contrato del agente es minúsculo: `load(onProgress)` y `chat(history,
// system, onToken) → string`. Todo lo demás es opcional. Este fichero traduce
// entre ese contrato y el motor.
//
// Cuatro detalles del contrato que no son obvios y que se pagan caros:
//   · El progreso se emite como STRING y la barra de carga le saca el «N %» con
//     un regex. Sin «%» la barra se queda indeterminada para siempre — le pasa
//     al proveedor de LiteRT y por eso su carga parece colgada.
//   · Los errores se le enseñan LITERALES al usuario. Se escriben para humanos.
//   · Las tool-calls no requieren nada: el agente busca un bloque ```tool en el
//     texto devuelto. No hay que filtrar los fences ni inventar un API.
//   · El `system` cambia en CADA turno porque lleva el contexto vivo dentro, así
//     que no sirve como clave de caché salvo por su prefijo.
//
// Y uno opcional: `chat()` admite un cuarto argumento, `{ signal, onPrefill }`,
// o la señal suelta (o un AbortController), que es como lo llamaban las apps
// antes de que existiera ese objeto. Un turno parado RECHAZA con un AbortError
// que trae en `.partial` lo que llegó a emitir, y `.porDescarga` a true si no
// lo paró quien llamaba sino una descarga (`unload()` o `configure()` a otro
// modelo); quien llame tiene que tratarlo como «parado» y no como «el modelo
// falló». `abort()` para lo que haya en marcha.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import { requestDevice, readback } from './device.js';
import { loadModel, createSession, errorParado, _setLote } from './model.js';

// Contextos a intentar, del pedido hacia abajo por mitades, con un suelo. El
// suelo NO es cero a propósito: por debajo de 2048 el modelo no puede ni con su
// propio prompt de sistema, así que abrir con 512 sería entregar algo que carga
// y no sirve, que es peor que decir que no cabe.
const CONTEXTO_MINIMO = 2048;
export function escaleraContexto(pedido) {
  const pasos = [];
  for (let c = Math.max(CONTEXTO_MINIMO, pedido | 0); c >= CONTEXTO_MINIMO; c = Math.floor(c / 2)) {
    pasos.push(c);
  }
  return pasos;
}
import { createTokenizer } from './tokenizer.js';
import { sample } from './sample.js';
import { parseGGUF } from './gguf.js';
// El registro va aparte para que la app pueda sondear disponibilidad sin
// importarse el motor entero. Ver la cabecera de registro.js.
import { MODELS, cachea } from './registro.js';

export let name = 'Elffuss Engine · WebGPU';

// El modelo por defecto del motor: LFM3 (ver registro.js).
let clave = 'lfm3-1.7b';
let ctxEfectivo = 0;   // el peldaño de la escalera en el que se abrio la sesion

// ── UN SOLO MOTOR PESADO POR NAVEGADOR ──────────────────────────────────────
// Cada pestaña cargaba su propia instancia sin enterarse de las demas. Con el
// modelo grande, cada una pedía los pesos enteros más su caché KV (que a
// contexto completo pesa más de la mitad que los pesos), peleandose por el mismo
// dispositivo, y el sintoma medido es brutal: el prefill cae a
// una decima parte y acaba yendo MAS LENTO que la generacion, que es imposible
// en un motor sano. Desde fuera no parece contencion, parece que el motor es
// malo.
//
// Esto es un CERROJO, no todavia una solucion: impide la segunda instancia
// pesada y lo dice con todas las letras. Compartir el motor de verdad —un
// SharedWorker con WebGPU dentro, que ya esta comprobado que funciona— es el
// paso siguiente y bastante mas grande.
//
// Solo para modelos GRANDES: dos instancias del pequeño caben de sobra y
// bloquearlas seria entorpecer sin motivo.
const CERROJO_DESDE = 2 * 1024 * 1024 * 1024;   // 2 GB de pesos
let soltarCerrojo = null;

async function tomarCerrojo(etiqueta) {
  // Sin la API (navegador viejo) no se bloquea nada: mas vale lento que roto.
  if (typeof navigator === 'undefined' || !navigator.locks) return true;
  if (soltarCerrojo) return true;                       // ya lo tenemos
  let decidir;
  const resultado = new Promise(r => { decidir = r; });
  navigator.locks.request('elffuss-motor-pesado', { ifAvailable: true }, (cerrojo) => {
    if (!cerrojo) { decidir(false); return; }            // lo tiene otra pestaña
    decidir(true);
    // Se mantiene tomado mientras viva la sesion: la promesa no se resuelve
    // hasta `unload()`.
    return new Promise(r => { soltarCerrojo = r; });
  }).catch(() => decidir(true));                         // si la API falla, no estorbar
  return resultado;
}
let modelo = null, sesion = null, tk = null, gpu = null, muestreo = null;
// ¿Este modelo espera un bloque de razonamiento en el prompt? Se DEDUCE de la
// plantilla que trae el propio fichero, no se cablea por nombre de modelo: así
// un GGUF sin razonamiento no recibe un bloque que no entiende, y uno nuevo que
// sí lo lleve funciona sin tocar nada.
let pensante = false;
let cargando = null;
let descargando = null;                    // la descarga en curso, si la hay: ver `unload`
// Las fichas del bloque de sistema que describe la ranura 'sistema' de la
// instantánea. `null` = no hay foto del sistema válida. Se guardan las FICHAS y
// no un hash para poder comparar una por una: un hash igual con contenido
// distinto contestaría de otra conversación sin dar ningún error.
let sistemaFoto = null;
let perdida = '';                          // motivo si el dispositivo se perdió por debajo

// Los tokens que el estado de la sesión YA ha ingerido: prompt del turno
// anterior más lo que se le realimentó al generar. Es la clave para no volver a
// procesar la conversación entera en cada turno. `null` = el estado no sirve y
// hay que reiniciar. Se invalida ANTES de tocar el estado y solo se vuelve a
// poner al terminar bien: si un turno revienta a medias, el siguiente reinicia.
let consumidos = null;

export function configure(key) {
  if (key && MODELS[key] && key !== clave) {
    clave = key;
    if (modelo) unload();                  // otro modelo: fuera el anterior
    return true;
  }
  return false;
}

// ¿Este modelo pide que se le hable en corto? El agente decide hoy por tamaño
// de contexto, y eso no basta: a este le CABE el prompt completo y aun así
// tarda catorce veces más con él. Se pregunta al proveedor en vez de deducirlo.
// Ningún modelo pide ya prompt compacto: recortarlo los dejaba sin saber hacer
// tareas. Se mantiene la función porque las apps la llaman con `?.()` y
// devolver false es la respuesta correcta, no un hueco.
export function prefiereCompacto() {
  return false;
}

export function models() {
  return Object.entries(MODELS).map(([id, m]) => ({ id, label: m.label }));
}

// Gancho SOLO para los tests: apunta el modelo activo a una URL servida en
// local. Sin esto, probar el proveedor exigiría bajar el modelo entero de
// Hugging Face en cada ejecución, que es la clase de test que nadie vuelve a correr.
export function __setModelForTest(url, bytes = 0) {
  MODELS[clave] = { ...MODELS[clave], url, bytes };
}

// Gancho SOLO para los tests: olvida el prefijo ingerido, de modo que el turno
// siguiente reinicie el estado como si fuera el primero. Sin esto, un banco que
// corre la MISMA conversación dos veces depende de una rama lateral —que el
// prompt de la segunda vuelta sea más corto que lo ya ingerido— para partir
// limpio, y entonces no estaría probando lo que dice probar.
export function __olvidaPrefijo() {
  consumidos = null;
  sistemaFoto = null;   // las DOS ranuras, o el banco mediría con la del sistema viva
}

// Handle de lectura por rangos. Primero OPFS (queda cacheado en disco entre
// visitas); si no está disponible, rangos HTTP directos — más lento en la
// segunda visita, pero funciona en vez de fallar.
async function abrirPorRangos(url, onProgress, cachear = true) {
  // ABSOLUTA siempre. El almacén puede delegar en el broker compartido, que vive
  // en OTRO origen: una URL relativa se resolvería contra el suyo y apuntaría a
  // un fichero que allí no existe.
  // El reparto se comprueba ANTES de normalizar: `new URL()` sobre un array
  // hace cosas raras en silencio en vez de fallar.
  // El reparto devuelve su propio `donde`: decir solo «2 servidores» ocultaba lo
  // único que hacía falta saber para juzgar el cacheo —si vino del almacén o de
  // la red—, y sin eso la prueba tenía que deducirlo por el reloj, que es
  // justamente el instrumento que ya engañó tres veces.
  if (Array.isArray(url)) return await rangosRepartidos(url, onProgress, cachear);
  try { url = new URL(url, location.href).href; } catch { /* sin location */ }
  if (cachear) {
    try {
      const store = await import('../runtime/model-store.js');
      // SIN guardián aquí. Lo hubo —45 s— de cuando el broker no tenía ninguno
      // y una petición sin respuesta dejaba la carga colgada para siempre. Ese
      // guardián ya vive DONDE corresponde: el SDK del broker corta por
      // inactividad (30 s rearmados con cada mensaje), que es la capa que sabe
      // distinguir «tarda» de «enmudeció».
      //
      // Mantenerlo fuera hacía daño: el plazo se agotaba en toda descarga real,
      // el motor caía a leer por red, y la del almacén seguía por detrás hasta
      // guardar. O sea que SÍ se cacheaba, pero por efecto secundario y con la
      // primera carga el doble de lenta. Un fallo que se disfraza de que todo
      // va bien es peor que uno que se ve.
      const ranged = await store.openRanged(url, onProgress);
      if (ranged) return { ranged, donde: 'disco (OPFS)' };
    } catch (e) {
      console.warn('[engine] almacén no disponible, se lee por red:', e.message);
    }
  }

  onProgress('Leyendo el modelo por partes desde la red…');
  const cabeza = await fetch(url, { headers: { Range: 'bytes=0-0' } });
  const cr = cabeza.headers.get('content-range');
  // Comprobar el 206 y no solo la cabecera: un servidor que ignora el Range
  // responde 200 con el fichero ENTERO, y aquí eso es el modelo entero en
  // memoria de golpe — la pestaña se muere sin decir por qué.
  if (cabeza.status !== 206 || !cr) {
    throw new Error('El servidor del modelo no admite descargas por partes, y el modelo es demasiado grande para bajarlo entero.');
  }
  const size = +cr.split('/')[1];
  return {
    donde: 'red (sin caché)',
    ranged: {
      size,
      // Se COMPRUEBA que vuelven los bytes pedidos. Un modelo grande son cientos
      // de peticiones, y basta UNA corta —o un 200 porque el servidor ignoró el Range— para que
      // esos bytes de menos entren como PESOS: sin error, sin traza, modelo
      // «cargado» y luego mudo o delirante. En localhost no pasa casi nunca y
      // por internet sí, o sea que el fallo solo aparecía servido de verdad: el
      // peor reparto posible entre dónde se prueba y dónde se rompe.
      //
      // gguf.js ya hacía esta comprobación para la cabecera («lectura corta: N
      // de M bytes»). La disciplina estaba escrita en un camino y faltaba en el
      // otro, que es donde van los gigabytes.
      async slice(offset, length) {
        if (length <= 0) return new ArrayBuffer(0);
        const fin = Math.min(offset + length, size) - 1;
        const quiero = fin - offset + 1;
        let ultimo = '';
        for (let intento = 0; intento < 3; intento++) {
          try {
            const r = await fetch(url, { headers: { Range: `bytes=${offset}-${fin}` } });
            // Un 200 aquí significa que el servidor ignoró el Range y viene el
            // fichero ENTERO, todo a memoria. Se rechaza antes de leerlo.
            if (r.status !== 206) { ultimo = `respondió ${r.status} y no 206`; continue; }
            const buf = await r.arrayBuffer();
            if (buf.byteLength === quiero) return buf;
            ultimo = `devolvió ${buf.byteLength} bytes de ${quiero}`;
          } catch (e) { ultimo = String(e && e.message || e); }
        }
        throw new Error(`El servidor del modelo devolvió datos incompletos y los pesos ` +
          `quedarían corruptos (bytes ${offset}-${fin}: ${ultimo}). Vuelve a intentarlo.`);
      },
    },
  };
}

// Handle por rangos sobre un modelo REPARTIDO entre varios hosts. Cada
// fragmento vive en su propio subdominio, y eso no es capricho: la cuota de
// almacenamiento del navegador es POR ORIGEN —cada origen cuenta solo lo
// suyo—, así que repartir multiplica el techo. El modelo grande no cabía fiable
// en un origen, y en dos entra holgado.
//
// Hacia arriba se ve UN fichero: mismo {size, slice} que el resto del motor.
async function rangosRepartidos(fragmentos, onProgress, cachear = true) {
  onProgress('Leyendo el modelo repartido entre servidores…');
  // Primero el ALMACÉN: cada fragmento lo guarda el broker de su propio origen,
  // troceado en partes porque el navegador corta un fichero suelto demasiado
  // grande. Así la segunda visita no vuelve a pedirlo; sin esto, la descarga
  // entera en cada sesión.
  // Lo que el almacén SÍ consiguió, por URL. Vale aunque falte algo: lo que hay
  // aquí no se vuelve a pedir por red.
  //
  // Antes esto era todo-o-nada: un `throw` al primer fragmento que fallara, y el
  // `catch` se bajaba por red los DOS, incluido el que ya estaba entero en
  // disco, o sea el doble de descarga en cada sesión que fallaba. El desperdicio
  // no estaba en el broker: estaba aquí.
  const delAlmacen = new Map();
  if (cachear) {
    try {
      const { getSharedModel } = await import('../runtime/model-broker.js');
      for (const [i, f] of fragmentos.entries()) {
        onProgress(`Recuperando el modelo · parte ${i + 1} de ${fragmentos.length}…`);
        try {
          const blob = await conLatido(latir => getSharedModel(f.url, m => {
            latir();
            if (m?.total) onProgress(`Descargando el modelo · parte ${i + 1}/${fragmentos.length} · ` +
              `${Math.round(m.loaded / m.total * 100)} %`);
          }), 60000, 'el almacén compartido');
          // El tamaño se comprueba igual que antes; lo que cambia es que un
          // fragmento malo ya no arrastra a los buenos.
          if (blob && (!f.bytes || blob.size === f.bytes)) delAlmacen.set(f.url, blob);
          else console.warn('[engine] fragmento incompleto en el almacén:', f.url);
        } catch (e) {
          console.warn('[engine] el almacén no dio', f.url, '→', e.message);
        }
      }
      if (delAlmacen.size === fragmentos.length) {
        return {
          ranged: desdeBlobs(fragmentos.map(f => delAlmacen.get(f.url))),
          donde: `almacén compartido · ${fragmentos.length} servidores`,
        };
      }
    } catch (e) {
      // Que el almacén falle NO puede impedir usar el modelo: se cae a leerlo
      // del servidor, que es más lento pero funciona.
      console.warn('[engine] almacén no disponible, se lee por red:', e.message);
    }
  }
  // NO paralelizar estas lecturas. Parece la optimización obvia —cientos de
  // peticiones una detrás de otra para cargar el 27B— y no lo es: varios flujos
  // a la vez no bajan antes que uno, porque el cuello es el enlace. La única
  // palanca real sobre el tiempo de carga es no volver a bajarlo: el almacén
  // de arriba.
  const trozos = [];
  let size = 0;
  for (const f of fragmentos) {
    // Si el almacén dio ESTE fragmento, se lee de disco y no se toca la red:
    // ni la sonda de tamaño (que ya lo sabemos) ni los rangos.
    const blob = delAlmacen.get(f.url);
    if (blob) { trozos.push({ url: f.url, desde: size, bytes: blob.size, blob }); size += blob.size; continue; }
    const cab = await fetch(f.url, { headers: { Range: 'bytes=0-0' } });
    const cr = cab.headers.get('content-range');
    if (cab.status !== 206 || !cr) {
      throw new Error('Un servidor del modelo no admite descargas por partes: ' + f.url);
    }
    const n = +cr.split('/')[1];
    // El tamaño DECLARADO se comprueba contra el real. Si un fragmento cambia
    // de tamaño, las fronteras se desplazan y todo lo que venga detrás se lee
    // corrido: basura sin un solo error.
    if (f.bytes && n !== f.bytes) {
      throw new Error(`Un fragmento del modelo mide ${n} y se esperaban ${f.bytes}.`);
    }
    trozos.push({ url: f.url, desde: size, bytes: n });
    size += n;
  }

  const cual = abs => {                          // búsqueda binaria del fragmento
    let lo = 0, hi = trozos.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (trozos[m].desde <= abs) lo = m; else hi = m - 1; }
    return lo;
  };
  const pedir = async (t, desde, cuanto) => {
    // Fragmento que vino del almacén: se corta el Blob, sin red.
    if (t.blob) return await t.blob.slice(desde, desde + cuanto).arrayBuffer();
    const fin = desde + cuanto - 1;
    for (let intento = 0; intento < 3; intento++) {
      const r = await fetch(t.url, { headers: { Range: `bytes=${desde}-${fin}` } });
      if (r.status !== 206) continue;
      const buf = await r.arrayBuffer();
      if (buf.byteLength === cuanto) return buf;
    }
    throw new Error(`Datos incompletos de ${t.url} (bytes ${desde}-${fin}).`);
  };

  return {
   // Honesto cuando es mixto: decir «red» con la mitad saliendo de disco
   // escondería justo lo que acabamos de arreglar.
   donde: delAlmacen.size
     ? `mixto · ${delAlmacen.size} del almacén y ${fragmentos.length - delAlmacen.size} por red`
     : `red · ${fragmentos.length} servidores`,
   ranged: {
    size,
    async slice(offset, length) {
      // Math.trunc y no |0: esto pasa de 2^31 y el bitwise trunca a 32 bits.
      const from = Math.max(0, Math.min(size, Math.trunc(offset) || 0));
      const want = Math.max(0, Math.min(size - from, Math.trunc(length) || 0));
      if (want === 0) return new ArrayBuffer(0);
      const i0 = cual(from), d0 = from - trozos[i0].desde;
      if (d0 + want <= trozos[i0].bytes) return await pedir(trozos[i0], d0, want);
      // A caballo de dos servidores. Pasa poco, pero pasa: los tensores no se
      // alinean con la frontera del reparto.
      const out = new Uint8Array(want);
      let puesto = 0;
      for (let i = i0; i < trozos.length && puesto < want; i++) {
        const d = i === i0 ? d0 : 0;
        const cuanto = Math.min(trozos[i].bytes - d, want - puesto);
        if (cuanto <= 0) continue;
        out.set(new Uint8Array(await pedir(trozos[i], d, cuanto)), puesto);
        puesto += cuanto;
      }
      return out.buffer;
    },
   },
  };
}

// Handle por rangos sobre fragmentos YA guardados en disco. Blob.slice() es
// perezoso: no se traga los gigabytes en memoria.
function desdeBlobs(ficheros) {
  const inicio = []; let size = 0;
  for (const f of ficheros) { inicio.push(size); size += f.size; }
  const cual = abs => { let lo = 0, hi = inicio.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (inicio[m] <= abs) lo = m; else hi = m - 1; } return lo; };
  return {
    size,
    async slice(offset, length) {
      const from = Math.max(0, Math.min(size, Math.trunc(offset) || 0));
      const want = Math.max(0, Math.min(size - from, Math.trunc(length) || 0));
      if (want === 0) return new ArrayBuffer(0);
      const i0 = cual(from), d0 = from - inicio[i0];
      if (d0 + want <= ficheros[i0].size) return await ficheros[i0].slice(d0, d0 + want).arrayBuffer();
      const out = new Uint8Array(want); let puesto = 0;
      for (let i = i0; i < ficheros.length && puesto < want; i++) {
        const d = i === i0 ? d0 : 0;
        const cuanto = Math.min(ficheros[i].size - d, want - puesto);
        if (cuanto <= 0) continue;
        out.set(new Uint8Array(await ficheros[i].slice(d, d + cuanto).arrayBuffer()), puesto);
        puesto += cuanto;
      }
      return out.buffer;
    },
  };
}

// Permite apuntar a un GGUF propio en vez del de Hugging Face. Sirve para
// desarrollo (servirlo en local) y para quien quiera autoalojarlo o probar su
// propia cuantización sin tocar el código.
function modeloActivo() {
  const m = MODELS[clave];
  let url = null;
  // ?engine-url=… gana sobre el ajuste guardado: así se puede compartir un
  // enlace que apunta a un modelo concreto sin tocar nada.
  try { url = new URL(location.href).searchParams.get('engine-url'); } catch { /* — */ }
  if (url) { try { localStorage.setItem('elffuss.engine.url', url); } catch { /* — */ } }
  if (!url) { try { url = localStorage.getItem('elffuss.engine.url'); } catch { /* — */ } }
  return url ? { ...m, url, bytes: 0 } : m;   // bytes 0: no comparar tamaños
}

// Un fallo de una dependencia debe DEGRADAR, no colgar. El almacén compartido
// delega en un broker cross-origin cuya petición no lleva timeout: si el broker
// recibe la petición y nunca contesta, la promesa no se resuelve NUNCA y la
// carga se queda parada sin error, sin log y sin recuperación. Aquí se le pone
// un límite y se sigue por otro camino.
function conLimite(promesa, ms, queHacer) {
  return Promise.race([
    promesa,
    new Promise((_, no) => setTimeout(() => no(new Error(queHacer + ': sin respuesta en ' + ms + ' ms')), ms)),
  ]);
}

// Igual, pero el reloj se REINICIA con cada señal de vida. Un tope total no
// sirve para algo que tarda lo que tarda una descarga: con un tope total sobre
// el almacén, una descarga normal agotaba siempre el plazo, caía a leer por red
// y nunca llegaba a guardarse. Parecía funcionar solo cuando ya estaba
// cacheado de antes, que es el peor disfraz posible.
//
// Lo que hay que detectar es que el otro lado ENMUDEZCA, no que tarde.
function conLatido(hacer, ms, queHacer) {
  let temporizador, vivo = true;
  return new Promise((si, no) => {
    const rearmar = () => {
      clearTimeout(temporizador);
      if (!vivo) return;
      temporizador = setTimeout(() => { vivo = false; no(new Error(queHacer + ': sin señales en ' + ms + ' ms')); }, ms);
    };
    rearmar();
    hacer(rearmar).then(v => { vivo = false; clearTimeout(temporizador); si(v); },
                        e => { vivo = false; clearTimeout(temporizador); no(e); });
  });
}

export async function load(onProgress = () => {}) {
  // Una descarga a medias se espera ANTES de mirar `sesion`: `configure()`
  // descarga sin esperar, y sin esto un `load()` justo detrás vería la sesión
  // vieja todavía puesta, contestaría «ya está cargado» y la descarga la
  // destruiría un instante después.
  if (descargando) await descargando;
  if (sesion) return;                      // idempotente: nunca dos motores
  if (cargando) return cargando;           // dos llamadas a la vez: una carga
  cargando = (async () => {
    const m = modeloActivo();
    try {
      onProgress('Encendiendo la GPU…');
      perdida = '';
      gpu = await requestDevice({ label: 'elffuss-engine' });

      // El device se comprueba al nacer, pero también se puede morir DESPUÉS: el
      // proceso gráfico del navegador se reinicia y se lleva por delante los
      // buffers, y con gigas de pesos dentro eso pasa de verdad. Sin enterarse,
      // lo siguiente que ocurre es un forward que devuelve ceros o una promesa
      // que no vuelve nunca: un cuelgue sin causa visible. Aquí se entera, y así
      // el siguiente turno puede decir qué pasó en vez de quedarse mudo.
      gpu.device.lost.then(info => {
        // El estado se anula SIEMPRE, tanto si el dispositivo se cayó solo como si
        // lo tiramos nosotros. Antes la rama 'destroyed' salía sin registrar nada y
        // —lo grave— sin anular `sesion` ni `modelo`: la app se quedaba creyendo
        // que seguía cargada sobre un dispositivo muerto, y el síntoma era
        // «listo» y después silencio, sin un solo error. Un manejador que solo
        // ve la muerte accidental y no la deliberada es un punto ciego que
        // cuesta una tarde de diagnóstico cada vez que aparece.
        const nuestra = info.reason === 'destroyed';
        // `sistemaFoto` también: sus buffers se van con la sesión, y creer que
        // hay foto cuando ya no existe restauraría basura sin dar error.
        sesion = null; modelo = null; consumidos = null; sistemaFoto = null;
        if (nuestra) { console.info('[engine] GPU liberada por nosotros (destroyed)'); return; }
        perdida = 'Se ha perdido la GPU con el modelo cargado' +
          (info.message ? ` (${info.message})` : '') + '.';
        console.warn('[engine] ' + perdida);
      });

      // ── El cerrojo, ANTES DE MOVER UN BYTE ───────────────────────────────
      // ⚠️ Esto estaba al final, justo antes de reservar contexto, con un
      // comentario que decía «el cerrojo se pide ANTES de reservar contexto».
      // Era verdad y era inútil: para entonces la pestaña ya se había bajado el
      // modelo entero y lo había subido al dispositivo. O sea que la segunda pestaña
      // pagaba la carga completa de un modelo de este tamaño para que al final
      // se le dijera que no. Peor todavía: durante toda esa carga SÍ había dos
      // copias peleándose por el dispositivo, que es exactamente lo que el cerrojo
      // existe para impedir.
      //
      // Pedirlo aquí no arriesga quedárselo colgado: cualquier excepción de
      // esta carga pasa por el `catch` de abajo, que llama a `unload()`, y
      // `unload()` suelta el cerrojo.
      if ((m.bytes || 0) >= CERROJO_DESDE && globalThis.__ELFFUSS_SIN_CERROJO !== true) {
        if (!(await tomarCerrojo(m.label))) {
          throw new Error(
            `Ya hay otra pestaña de Elffuss con ${m.label} cargado. Dos copias de ` +
            `un modelo de este tamaño no caben en la GPU a la vez y las dos van ` +
            `diez veces más lentas. Cierra la otra pestaña y vuelve a elegirlo aquí.`);
        }
      }

      // Avisar ANTES de bajar, no después: el usuario que no tiene sitio se
      // entera ahora y no al final de una descarga entera.
      const apretado = cachea(m) ? await faltaSitio(m.bytes) : null;
      if (apretado) {
        throw new Error(`${m.label} necesita unos ${GiB(apretado.necesita)} GiB y el navegador ` +
          `solo puede guardar ${GiB(apretado.libre)} GiB más. Libera espacio en disco.`);
      }

      const { ranged, donde } = await abrirPorRangos(m.url, onProgress, cachea(m));
      if (m.bytes && ranged.size !== m.bytes) {
        console.warn(`[engine] el modelo mide ${ranged.size} y se esperaban ` +
          `${m.bytes}: el fichero de origen ha cambiado`);
      }

      // La cabecera primero: dice cuánto pesa cada tensor y permite dar
      // progreso REAL durante la subida, que es la parte larga.
      onProgress('Leyendo el índice del modelo… 0 %');
      const info = await parseGGUF(ranged);
      tk = createTokenizer(info.meta);
      // El lote del prefill es del MODELO: 32 por defecto (el bucle recurrente
      // del SSM corre entero dentro de un despacho y un lote largo cuelga el
      // escritorio); el registro lo sube donde no hay ese bucle.
      _setLote(m.lote || 32);
      pensante = /<think>/.test(String(info.meta['tokenizer.chat_template'] || ''));
      formato = formatoDePlantilla(info.meta['tokenizer.chat_template'], tk, info.meta['general.architecture']);

      // La etiqueta cambia según de dónde salgan los pesos, y no es cosmética.
      // Sin caché, esta fase NO es solo «subir»: es bajar los gigabytes y
      // subirlos según llegan, y el tiempo se lo lleva casi todo la red. Decir
      // solo «subiendo» mientras alguien espera a que baje el modelo entero
      // hace que parezca colgado en el paso equivocado.
      const etiqueta = donde.startsWith('red')
        ? 'Descargando el modelo y subiéndolo a la GPU'
        : 'Subiendo el modelo a la GPU';
      let ultimo = -1;
      modelo = await loadModel(ranged, gpu.device, {
        debug: false,
        // Ancho de subgrupo del dispositivo: activa la reducción por subgrupos
        // del matmul (más rápida que la de árbol). 0 si el dispositivo no los tiene.
        subgroupSize: gpu.subgroupSize || 0,
        onProgress: (hechos, total) => {
          const pct = Math.floor(hechos / total * 100);
          if (pct === ultimo) return;      // no inundar la UI con el mismo %
          ultimo = pct;
          onProgress(`${etiqueta} · ${pct} %`);
        },
      });

      // El contexto se pide ENTERO al abrir la sesión: la caché KV se reserva de
      // una vez, no crece. A 64k eso es mucha memoria ANTES de que el usuario
      // escriba nada, y en un dispositivo con menos memoria la reserva falla.
      // Sin esta escalera, ese fallo deja al usuario sin modelo cuando lo
      // que quería era un contexto menor — un contexto de 16k sigue siendo
      // ocho veces el de antes y no se parece en nada a no tener modelo.
      //
      // No se adivina por `navigator.deviceMemory` ni por el tamaño del
      // adaptador: se INTENTA y se baja al fallar, que es el único método que no
      // depende de un dato declarado que no todos los navegadores exponen.
      // (el aviso de espacio se da ANTES de descargar; ver `faltaSitio`)
      let pedido = m.ctx, ultimoFallo = null;
      for (const c of escaleraContexto(m.ctx)) {
        try { sesion = createSession(modelo, { contextLen: c }); pedido = c; break; }
        catch (e) { ultimoFallo = e; }
      }
      if (!sesion) throw ultimoFallo || new Error('no se pudo reservar ningún contexto');
      ctxEfectivo = pedido;
      if (pedido !== m.ctx) {
        console.warn(`[engine] ${m.label}: no cupo un contexto de ${m.ctx}, se abre con ${pedido}`);
      }
      name = 'Elffuss Engine · ' + m.label;
      console.log(`[engine] ${m.label} listo desde ${donde} · ${sesion.state.summary()}`);
      onProgress('Modelo listo · 100 %');
    } catch (e) {
      await unload();
      // El kernel enseña este mensaje TAL CUAL al usuario.
      throw new Error(mensajeHumano(e, m));
    } finally {
      cargando = null;
    }
  })();
  return cargando;
}

// Comprueba que quepa ANTES de empezar a bajar gigas. Sin esto, el disco lleno
// se manifiesta como un `Failed to fetch` a mitad de descarga: un mensaje que
// culpa a la red de un problema de espacio, después de hacer esperar al usuario.
// Visto de verdad con el modelo grande: sin sitio para guardarlo, la descarga
// muere y el error que llega arriba habla de red.
//
// `estimate()` no existe en todos los navegadores y su cuota es orientativa, así
// que esto solo AVISA cuando está claro que no cabe; nunca impide intentarlo por
// no saberlo. Un falso negativo deja al usuario donde estaba; un falso positivo
// le negaría un modelo que sí cabía.
async function faltaSitio(bytes) {
  try {
    if (!navigator.storage?.estimate || !bytes) return null;
    const { quota, usage } = await navigator.storage.estimate();
    if (!quota) return null;
    const libre = quota - (usage || 0);
    // Margen del 5 %: la cuota es una estimación y escribir justo al borde
    // falla igual, solo que más tarde y peor.
    return libre < bytes * 1.05 ? { libre, necesita: bytes, quota } : null;
  } catch { return null; }
}

const GiB = (n) => (n / 1024 ** 3).toFixed(1);

function mensajeHumano(e, m) {
  const t = String(e?.message || e);
  // `Failed to fetch` es lo que sale cuando la escritura al almacenamiento
  // revienta a media descarga, y leerlo como un problema de red manda al
  // usuario a mirar su wifi en vano.
  if (/Failed to fetch|NetworkError|QuotaExceeded|exceed.*quota/i.test(t)) {
    return `No se pudo descargar ${m.label}. Suele ser falta de espacio en disco ` +
      `—ocupa unos ${GiB(m.bytes || 0)} GiB— o una conexión que se ha cortado. ` +
      'Libera espacio y vuelve a intentarlo.';
  }
  if (/límite de binding|maxStorageBufferBindingSize/i.test(t)) {
    return `Este navegador no deja crear buffers lo bastante grandes para ${m.label}. ` +
      'En iPhone el límite es mucho menor que en escritorio.';
  }
  if (/sin WebGPU|no hay adaptador/i.test(t)) {
    return 'Este navegador no tiene WebGPU disponible. Hace falta Chrome, Edge o Safari 18+.';
  }
  if (/out-of-memory|OutOfMemory/i.test(t)) {
    return `No hay memoria de GPU suficiente para ${m.label} en este equipo.`;
  }
  return `No se pudo cargar ${m.label}: ${t}`;
}

// Descargar empieza por PARAR. Antes se destruían los buffers con un turno en
// marcha —la app llama aquí justo cuando la memoria va justa, que
// es cuando más probable es que haya uno— y el forward seguía grabando sobre
// memoria ya liberada. Ahora se para el turno, se espera a que suelte el motor
// y solo entonces se destruye.
//
// Se guarda la promesa en `descargando` para que `load()` y los turnos que
// lleguen mientras tanto la esperen en vez de ver la sesión vieja.
export async function unload() {
  if (!descargando) descargando = descargar().finally(() => { descargando = null; });
  return descargando;
}

async function descargar() {
  // Con su propio motivo: el turno parado así no lo pidió quien lo lanzó, y
  // el error tiene que decirlo (ver «POR QUÉ se paró»).
  try { await pararTodos('descarga'); } catch { /* parar nunca impide descargar */ }
  pensante = false;
  formato = formatoDePlantilla('', null);
  try { sesion?.destroy?.(); } catch { /* mejor esfuerzo */ }
  try {
    for (const { buf } of modelo?.weights?.values?.() || []) buf.destroy();
  } catch { /* mejor esfuerzo */ }
  try { gpu?.device?.destroy?.(); } catch { /* mejor esfuerzo */ }
  sesion = null; modelo = null; tk = null; gpu = null;
  // Soltar el cerrojo: si no, la pestaña seguiria reservando el modelo grande
  // despues de descargarlo y ninguna otra podria cargarlo.
  try { soltarCerrojo?.(); } finally { soltarCerrojo = null; }
  // Y el rastro del prefijo, o el siguiente turno creería que puede reutilizar
  // contra un estado que ya no existe. No es teórico: `configure()` llama aquí
  // al cambiar de modelo y la app llama a `unload()` cuando el
  // heap se dispara. Un `consumidos` que sobreviva a eso describe tokens que la
  // sesión NUEVA no ha ingerido, y la reutilización saldría adelante contra un
  // estado limpio: texto plausible, ningún error, imposible de atribuir.
  //
  // Y `sistemaFoto` por la misma razón, agravada: `destroy()` libera los
  // buffers sombra de las dos ranuras, así que una foto del sistema que
  // sobreviva a esto apunta a memoria liberada.
  consumidos = null;
  sistemaFoto = null;
}

// {temperature, seed, p, k, maxTokens} o null para volver a voraz. Lo usa el
// jurado de rlm.js para pedir varias respuestas distintas del mismo prompt.
//
// Los campos se copian uno a uno en vez de propagar el objeto: así una clave
// desconocida no se cuela hasta el muestreador, donde cambiaría la salida en
// silencio. `maxTokens` es el tope de la respuesta.
export function setSampling(opts) {
  if (!opts) { muestreo = null; return; }        // null = volver al del registro (o a voraz)
  // FUSIONA sobre lo que hubiera, no reemplaza. Lo más natural que hace quien
  // usa esto es acotar solo la longitud —setSampling({maxTokens: 120})— y con
  // la versión que reemplazaba, esa llamada devolvía en silencio la temperatura
  // y la semilla a sus valores por defecto. Cambiar el muestreo entero cuando
  // te han pedido un tope de longitud es de las cosas que cuestan horas de
  // encontrar, porque no falla: sale otra respuesta y ya.
  //
  // Se sigue copiando campo a campo en vez de propagar el objeto: una clave
  // desconocida no debe colarse hasta el muestreador, donde cambiaría la salida
  // sin que nadie lo haya pedido.
  // Se fusiona sobre lo que haya o, si no hay nada, sobre el muestreo que el
  // REGISTRO declara para el modelo: acotar la longitud de un 27B no debe
  // devolverlo a voraz en silencio.
  const a = muestreo || muestreoDelRegistro() || {};
  muestreo = {
    temperature: opts.temperature ?? a.temperature ?? 0,
    topP: opts.p ?? a.topP ?? 1,
    topK: opts.k ?? a.topK ?? 0,
    minP: opts.minP ?? a.minP ?? 0,
    presence: opts.presence ?? a.presence ?? 0,
    repeatPenalty: opts.repeatPenalty ?? a.repeatPenalty ?? 1.1,
    seed: opts.seed ?? a.seed ?? 1,
    maxTokens: opts.maxTokens ?? a.maxTokens ?? 0,
    // true/false lo fuerza; null, lo que diga el registro del modelo.
    // `razonar: null` explícito vuelve a lo del registro; sin la clave, se conserva.
    razonar: 'razonar' in opts ? (typeof opts.razonar === 'boolean' ? opts.razonar : null) : (a.razonar ?? null),
  };
}

// El gestor del historial del núcleo de agente. Si no está (motor usado suelto,
// fuera de Elffuss), se cae a la cola reciente en vez de fallar.
async function ajustaHistorial(history, presupuesto) {
  try {
    const { packHistoryAsync } = await import('../context.js');
    return await packHistoryAsync(history, presupuesto);
  } catch { return history.slice(-12); }
}

// ChatML, que es lo que espera esta familia de modelos. La plantilla del GGUF
// es Jinja y no se interpreta aquí: se construye el mismo formato a mano.
// El bloque de razonamiento VACÍO que abre la plantilla. Se saca a una constante
// porque es exactamente lo que hay que poder separar: entra en el estado al
// generar, pero NO se reproduce cuando ese turno pasa a ser historia, así que
// mientras esté dentro, lo ingerido no puede ser prefijo del prompt siguiente.
// Medido: el desfase era de 5 tokens constantes, y son estos.
const PENSAR_QWEN = '<think>\n\n</think>\n\n';
// …y el bloque ABIERTO, para cuando el modelo SÍ razona: la plantilla termina en
// «<think>\n» y el modelo escribe su razonamiento, el «</think>» y la respuesta.
const ABRIR_QWEN = '<think>\n';

// LOS DETALLES DEL FORMATO SALEN DE LA PLANTILLA DEL FICHERO. Todas estas
// familias son ChatML, pero no escriben igual los bordes, y un borde distinto
// no peta: deja al modelo en un estado que no vio al entrenar.
//   bos        LFM3 empieza por `{{- bos_token -}}` («<|startoftext|>») y no lo
//              añade el tokenizador (add_bos_token = false): o lo pone el
//              prompt, o no está.
//   cerrado    «<think></think>» pegado en LFM3; con saltos de línea en Qwen.
//   abrir      «<think>» en LFM3; «<think>\n» en Qwen.
//   historial  LFM3 escribe «<think></think>» delante de CADA turno pasado del
//              asistente. Además de ser lo que vio al entrenar, hace que el
//              turno generado sin razonar y el mismo turno vuelto a escribir
//              como historia sean IGUALES, y el prefijo se pueda reutilizar.
//
// ⚠️ El BOS por plantilla solo se aplica a las arquitecturas VALIDADAS con él.
// MiniCPM5 (arch `llama`) también empieza su plantilla por `bos_token` con
// add_bos_token = false, y hoy corre SIN su «<s>». Añadírselo cambia la salida
// de un modelo ya ofrecido, así que se valida aparte antes de tocarlo.
const BOS_POR_PLANTILLA = new Set(['lfm3']);
let formato = { bos: '', cerrado: PENSAR_QWEN, abrir: ABRIR_QWEN, historial: '' };
function formatoDePlantilla(plantilla, tok, arch = 'lfm3') {
  const t = String(plantilla || '');
  const f = { bos: '', cerrado: PENSAR_QWEN, abrir: ABRIR_QWEN, historial: '' };
  if (BOS_POR_PLANTILLA.has(arch) && /^\s*\{\{-?\s*bos_token\s*-?\}\}/.test(t) && tok?.bosId !== undefined) f.bos = tok.tokenText(tok.bosId);
  if (t.includes('"<think></think>"')) { f.cerrado = '<think></think>'; f.abrir = '<think>'; f.historial = '<think></think>'; }
  return f;
}

// ¿Razona ESTE turno? Solo si el modelo es pensante (lo dice su plantilla) y:
//   · `setSampling({ razonar: true|false })` lo fuerza, o
//   · si nadie lo pide, lo decide el registro (`razona: true` en el modelo).
// Una llamada que acota la respuesta en corto (`maxTokens` explícito por debajo
// de RAZONAR_MIN, como el jurado de rlm.js) no razona: con 8 o 120 tokens el
// razonamiento se comería el turno entero y saldría vacío.
//
// Por qué existe: en tareas de agente, MiniCPM5-2B sin razonar se paraba al
// segundo paso sin tocar nada, y razonando sí avanzaba. Con el bloque siempre
// cerrado, un modelo así casi no podía trabajar.
const RAZONAR_MIN = 1024;

// El muestreo que el registro declara para el modelo activo (la ficha de su
// fabricante), o null si no declara ninguno. Se usa cuando la app no ha fijado
// otro con setSampling. Sin esto todo modelo iba en voraz con penalización de
// repetición 1,1, que no es lo que recomienda ningún fabricante: el 27B, por
// ejemplo, pide 0,7 / 0,8 / k20 / presence 1,5 sin razonar.
function muestreoDelRegistro() {
  const m = MODELS[clave]?.muestreo;
  return m ? { ...m } : null;
}
function muestreoEfectivo() { return muestreo || muestreoDelRegistro(); }
const PRESUPUESTO_RAZONANDO = 4000;     // si el registro no dice otro
function razonaEsteTurno() {
  if (!pensante) return false;
  const m = muestreoEfectivo();
  const pedido = m?.razonar;
  if (pedido === false) return false;
  if (m?.maxTokens && m.maxTokens < RAZONAR_MIN && pedido !== true) return false;
  return pedido === true || !!MODELS[clave]?.razona;
}
// Tope de la respuesta. Sin razonar, lo de siempre: 512 como mucho. Razonando,
// el presupuesto del modelo (o el que pida quien llama): el razonamiento va
// DENTRO de esa cuenta, y con 512 MiniCPM ni llegaba a emitir la herramienta.
function topeRespuesta(razona) {
  const pedido = muestreoEfectivo()?.maxTokens || 0;
  if (razona) return pedido || MODELS[clave]?.presupuesto || PRESUPUESTO_RAZONANDO;
  // Sin razonar: 512, o el `tope` que declare el modelo (el 27B escribe ficheros
  // enteros en una respuesta, y 512 los cortaba).
  const techo = MODELS[clave]?.tope || 512;
  return Math.min(techo, pedido || techo);
}

// El bloque de sistema, aparte y con nombre: la ranura 'sistema' de la
// instantánea necesita saber cuántas fichas ocupa EXACTAMENTE, y escribir el
// formato dos veces sería garantizar que algún día divergen y la foto describe
// un punto que no es.
function bloqueSistema(system) { return `${formato.bos}<|im_start|>system\n${system}<|im_end|>\n`; }

function montarPrompt(system, mensajes, { conPensar = true } = {}) {
  let s = bloqueSistema(system);
  for (const m of mensajes) {
    const rol = m.role === 'assistant' ? 'assistant' : 'user';
    const pre = rol === 'assistant' && formato.historial && !String(m.content).startsWith('<think>') ? formato.historial : '';
    s += `<|im_start|>${rol}\n${pre}${m.content}<|im_end|>\n`;
  }
  s += '<|im_start|>assistant\n';

  // El bloque de razonamiento lo abre LA PLANTILLA, no el modelo. La del GGUF
  // termina en «<|im_start|>assistant\n» + «<think>\n» (razonando) o
  // «<think>\n\n</think>\n\n» (sin razonar). Parar en «assistant\n» deja al
  // modelo en un estado que NO vio en entrenamiento, y este contestaba con la
  // parada en el primer token: respuesta vacía, sin error, sin traza. Costó
  // encontrarlo porque parecía cosa de la cuantización.
  //
  // Se manda el bloque CERRADO —sin razonar— a propósito. Razonando escribiría
  // cientos de tokens antes de la primera palabra útil, y con este modelo cada
  // token se paga caro; además el cierre suelto que emitiría hay que recortarlo
  // después. Cerrado, va directo a la respuesta.
  if (conPensar && pensante) s += formato.cerrado;
  return s;
}

// Cuántos tokens de contexto tiene el modelo elegido. La app lo consulta ANTES
// de escribir el prompt para decidir qué le cabe decir: sin esto le mandaba las
// instrucciones enteras al 27B —512 de contexto— y el primer mensaje moría con
// la caché llena, después de descargar el modelo entero.
// Se puede preguntar sin haber cargado nada: es una propiedad del registro.
export function contextTokens() {
  return MODELS[clave]?.ctx || 0;
}

// ── MEDIDAS POR TURNO ───────────────────────────────────────────────────────
// Hasta ahora el motor no medía NADA: las pruebas cronometraban desde fuera
// envolviendo la llamada, y la velocidad de generación había que deducirla
// aparte. Eso significa que la cifra que ve el usuario y la que mide una prueba
// no podían ser la misma, y que nadie sabía a qué iba su propio equipo.
//
// Las dos velocidades se cuentan SEPARADAS porque son fenómenos distintos: el
// prefill procesa N tokens de golpe y la generación uno a uno, así que mezclarlas
// da un número que no significa nada. Y el primer paso del bucle ES el prefill:
// `paso === 0` mete el prompt entero y los siguientes un token cada uno.
//
// `reusados` no es decoración: con la caché de prefijo funcionando, los tokens
// PROCESADOS son muchos menos que los del prompt, y sin ese dato el tok/s de
// prefill parece un milagro o una mentira según cómo se mire.
//
// La medida NACE cuando el turno coge el motor con un modelo cargado y se
// CIERRA en todas sus salidas, con `enCurso` a false: si acaba bien; si lo
// paran, con `abortado`; si falla, con el mensaje en `error`. Un turno parado
// en la cola no llegó a coger el motor y no tiene medida: `ultimaMedida()`
// sigue siendo la del turno de delante, que puede estar todavía en marcha.
const ahora = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
let medida = null;
const oyentes = new Set();
const avisar = (m = medida) => { for (const cb of oyentes) { try { cb(m); } catch { /* un oyente roto no tumba la generación */ } } };

/** La medida del último turno (o el turno en curso). Null si no ha habido ninguno. */
export function ultimaMedida() { return medida; }

/**
 * Qué modelo está activo, para poder ETIQUETAR las medidas.
 *
 * Sin esto, una captura del velocímetro no se puede diagnosticar: llegó una con
 * el prefill mucho mas lento de lo esperable y no habia forma de saber si era
 * el modelo pequeño o el grande, que rinden en ordenes distintos. Una cifra de velocidad sin el modelo al
 * lado no significa nada.
 */
/**
 * El contexto con el que se ABRIO de verdad la sesion, que no tiene por que ser
 * el del registro: si no cabe, la escalera baja a la mitad y vuelve a probar.
 *
 * Esto se exporta porque el contexto decide cuanta cache KV se reserva —en el
 * modelo grande son gigas— y por tanto cuanta memoria del dispositivo queda para
 * trabajar. Dos sesiones del MISMO modelo pueden caer en peldaños distintos
 * segun la memoria libre de ese momento y rendir muy distinto, y hasta ahora
 * eso solo se decia en la consola: desde fuera, dos velocidades incomparables
 * parecian la misma medida.
 */
export function contextoEfectivo() { return ctxEfectivo; }

export function modeloEtiqueta() {
  const m = MODELS[clave];
  return m ? (m.label || clave) : clave;
}

/** Suscribe a las medidas. Avisa al acabar el prefill, en cada token y al cerrar el turno, acabe como acabe. Devuelve el cancelador. */
export function alMedir(cb) { oyentes.add(cb); return () => oyentes.delete(cb); }

// ── TURNOS DE UNO EN UNO, Y PARAR ───────────────────────────────────────────
// El motor tiene UN estado: la caché KV, el recurrente del SSM y las fotos del
// prefijo. Dos turnos a la vez lo pisarían el uno al otro sin dar un error —
// cada uno avanzando nPast sobre lo que escribió el otro—, así que `chat()` se
// pone a la cola detrás del turno anterior y solo empieza cuando ese suelta el
// motor. Dos llamadas a la vez dan lo mismo que dos seguidas.
//
// Parar es un AbortSignal, el mismo contrato que ya usan los otros proveedores
// de las apps: `opts.signal` en el cuarto argumento de `chat()`, o la señal
// suelta en esa misma posición. Un turno parado RECHAZA con un AbortError que
// lleva en `.partial` el texto limpio que llegó a emitir, y después de notarlo
// ya no llama a `onToken` ni a `onPrefill` ni una vez más.
//
// ⚠️ Y quien deja de esperar un turno tiene que PARARLO, con su señal o con
// `abort()`. Abandonar la promesa no suelta el motor: el turno sigue hasta el
// final y los siguientes se ponen a la cola detrás de él. Un plazo que le gana
// la carrera a `chat()` sin parar el turno no ahorra nada, lo convierte en un
// atasco: cada llamada siguiente gasta su propio plazo esperando en la cola.
//
// `ultimoTurno` se resuelve cuando el último turno encolado ha soltado el
// motor (o ha renunciado a cogerlo); nunca se rechaza. `turnosVivos` son los
// controladores de los turnos encolados o en marcha: es lo que corta `abort()`.
let ultimoTurno = Promise.resolve();
const turnosVivos = new Set();

// POR QUÉ se paró. Quien llama tiene que poder distinguir su propio «para» del
// que no pidió: `unload()` —la app cuando va justa de memoria, el botón de liberar el modelo—
// y `configure()` a otro modelo también paran el turno en marcha, y enseñarle
// al usuario «parado a petición» cuando no ha pedido nada es mentirle. El
// motivo va en el texto del error y en `.porDescarga`, que es lo que se mira
// sin leer texto. Gana la PRIMERA causa: un turno ya parado no cambia de motivo.
const PARADO_POR = {
  peticion: 'turno parado a petición',
  descarga: 'turno parado: se ha descargado el modelo',
};
const causaDe = new WeakMap();             // señal de un turno → por qué se paró
function parar(ctl, causa) {
  if (ctl.signal.aborted) return;
  causaDe.set(ctl.signal, causa);
  ctl.abort();
}
function errorDeParada(senal, partial) {
  const causa = causaDe.get(senal) || 'peticion';
  const e = errorParado(PARADO_POR[causa]);
  e.partial = partial;
  e.porDescarga = causa === 'descarga';
  return e;
}

// La señal, o algo que la lleve dentro (un AbortController). Se reconoce por la
// forma y no por la clase: una señal de otro reino —un iframe, un worker— no es
// `instanceof` del AbortSignal de esta página y aun así es una señal
// perfectamente válida.
function senalDe(x) {
  if (!x || typeof x !== 'object') return null;
  if (typeof x.aborted === 'boolean') return x;
  if (x.signal && typeof x.signal.aborted === 'boolean') return x.signal;
  return null;
}

// Espera a que el turno de delante suelte el motor, salvo que paren a este
// antes: entonces rechaza en el acto, sin esperar a nadie.
function esperarTurno(antes, senal) {
  return new Promise((si, no) => {
    if (senal.aborted) { no(errorParado()); return; }
    const fuera = () => no(errorParado());
    senal.addEventListener('abort', fuera, { once: true });
    antes.then(() => { senal.removeEventListener('abort', fuera); si(); });
  });
}

/**
 * Para el turno en marcha y los que esperan en la cola. Resuelve cuando el
 * motor está en reposo: ningún turno de los que había lo está usando, el
 * dispositivo ha terminado lo que tenía y el estado es coherente para el siguiente turno.
 */
export async function abort() {
  await pararTodos('peticion');
}

// Lo de `abort()`, con el motivo dicho: `descargar()` para por 'descarga'.
async function pararTodos(causa) {
  const hasta = ultimoTurno;
  for (const c of turnosVivos) { try { parar(c, causa); } catch { /* — */ } }
  await hasta;
}

/**
 * Un turno. `onToken` recibe cada ficha generada.
 *
 * `opts.onPrefill(hechos, total)` avisa del avance del PREFILL, que es la
 * espera larga y hasta ahora invisible: con un prompt de arnés en el modelo
 * grande es la espera más larga del turno y la pantalla no decía nada en todo
 * ese rato, y una espera sin señal se lee como «se ha colgado». Se llama por
 * lote terminado en el dispositivo, no por lote encolado.
 *
 * `opts.signal` para el turno: ver «TURNOS DE UNO EN UNO, Y PARAR».
 *
 * ⚠️ El cuarto argumento acepta TAMBIÉN un AbortSignal suelto, o un
 * AbortController: Elffuss Code lleva pasando la señal en esa posición desde
 * antes de que existiera este objeto de opciones, y el motor la ignoraba en
 * silencio. Romper esa llamada para estrenar un parámetro sería cambiar el
 * contrato por comodidad mía.
 */
export async function chat(history, system, onToken = () => {}, opts = {}) {
  const o = opts && typeof opts.aborted === 'boolean' ? { signal: opts } : (opts || {});
  const onPrefill = typeof o.onPrefill === 'function' ? o.onPrefill : null;
  const tLlamada = ahora();
  const externa = senalDe(o.signal);
  const ctl = new AbortController();
  const reenviar = () => parar(ctl, 'peticion');
  if (externa?.aborted) parar(ctl, 'peticion');
  else externa?.addEventListener?.('abort', reenviar, { once: true });
  turnosVivos.add(ctl);
  const antes = ultimoTurno;
  let soltar;
  ultimoTurno = new Promise((r) => { soltar = r; });
  try {
    try {
      await esperarTurno(antes, ctl.signal);
    } catch {
      // Parado mientras esperaba: no llegó a tocar el motor. Su hueco se suelta
      // cuando lo suelte el de delante; soltarlo ya dejaría colarse al de detrás
      // mientras el de delante todavía está trabajando.
      antes.then(soltar);
      soltar = null;
      const parado = errorDeParada(ctl.signal, '');
      console.log('[engine] turno PARADO en la cola: no llegó a empezar' +
        (parado.porDescarga ? ' · por descarga' : ''));
      throw parado;
    }
    return await turnoParable(history, system, onToken, { senal: ctl.signal, onPrefill, tLlamada });
  } finally {
    turnosVivos.delete(ctl);
    externa?.removeEventListener?.('abort', reenviar);
    soltar?.();
  }
}

// El turno con el motor ya para él solo. Si lo paran, deja todo listo para el
// siguiente ANTES de soltar el motor: el dispositivo en reposo, lo prestado devuelto al
// pozo y la medida cerrada como parada. El estado no hace falta tocarlo: todo
// turno empieza con `restaura()` o con `reset()`, las fotos que el turno va a
// pisar se invalidan antes de su primer forward, y cada foto se vuelve a apuntar
// solo con lo suyo ingerido entero (ver «LO QUE EL TURNO VA A PISAR»). Así un
// turno parado a medias nunca deja una foto que describa algo que no pasó.
async function turnoParable(history, system, onToken, opciones) {
  const { senal } = opciones;
  // `cerrar` lo pone `turno()` en cuanto nace la medida: es lo que la cierra
  // desde aquí cuando el turno sale por un error.
  const hecho = { fase: 'entrada', texto: '', prompt: 0, salida: null, cerrar: null };
  try {
    return await turno(history, system, onToken, opciones, hecho);
  } catch (e) {
    // Solo es «parado» si lo pidieron Y el error es el de parar. Un fallo de
    // verdad que coincida con un «para» se enseña como el fallo que es.
    if (!(senal.aborted && e?.name === 'AbortError')) {
      // Un fallo también cierra la medida, con el fallo dentro: si no, se
      // quedaría «en curso» para siempre, esperando tokens de un turno que ya
      // no existe.
      hecho.cerrar?.({ error: String(e?.message || e) });
      throw e;
    }
    try { await gpu?.device?.queue?.onSubmittedWorkDone?.(); } catch { /* sin dispositivo, nada que esperar */ }
    try { sesion?.recycle(); } catch { /* mejor esfuerzo */ }
    hecho.cerrar?.({ abortado: true });
    const parado = errorDeParada(senal, limpiarRazonamiento(hecho.texto));
    console.log(`[engine] turno PARADO en ${hecho.fase} · prompt ${hecho.prompt} tokens · ` +
      `nPast ${sesion?.state?.nPast ?? '-'} · salieron ${hecho.salida ? hecho.salida.length : 0} tokens · ` +
      `devuelto ${parado.partial.length} car.` + (parado.porDescarga ? ' · por descarga' : ''));
    throw parado;
  }
}

async function turno(history, system, onToken, { senal, onPrefill, tLlamada }, hecho) {
  // Un turno parado antes de empezar no toca nada: ni el estado ni la foto.
  if (senal.aborted) throw errorParado();
  // Uno que llegó con una descarga en marcha la espera: cuando le toque, el
  // modelo ya no estará y lo dirá con el error de siempre. Se espera DESPUÉS de
  // mirar la señal y no antes: la descarga para a todos los turnos que había
  // antes que ella y espera a que suelten el motor, así que uno de esos
  // esperándola a ella se quedaría colgado para siempre.
  if (descargando) {
    await descargando;
    if (senal.aborted) throw errorParado();
  }
  hecho.fase = 'preparación';
  // Antes que nada: si el dispositivo se perdió, decirlo con esas palabras. Si no, el
  // usuario ve «El modelo todavía no está cargado» después de haber esperado a
  // que cargara, que es la peor manera de contar lo que ha pasado.
  if (perdida) throw new Error(perdida + ' Vuelve a elegirlo en el selector para cargarlo otra vez.');
  if (!sesion) throw new Error('El modelo todavía no está cargado.');

  // ── LA MEDIDA NACE AQUÍ, con el turno ya en el motor ───────────────────────
  // Antes nacía con la cabeza ya ingerida, y eso tenía dos agujeros: un turno
  // parado en la preparación o en el prefill no llegaba a tenerla, así que
  // `ultimaMedida()` seguía dando la del turno ANTERIOR como si fuera la suya y
  // nadie recibía aviso; y uno que fallaba a media respuesta se quedaba «en
  // curso» para siempre. Ahora se cierra en todas las salidas: al final de esta
  // función si acaba bien, y en `turnoParable` si lo paran o falla. Los campos
  // que aún no se saben salen a cero y se rellenan según se conocen.
  const m = medida = {
    promptTokens: 0, procesados: 0, reusados: 0,
    prefillMs: 0, prefillTokS: 0,
    genTokens: 0, genMs: 0, genTokS: 0,
    ttftMs: 0, totalMs: 0, enCurso: true, abortado: false, error: null,
  };
  let msForward = 0, msReadback = 0, msSample = 0, msDecode = 0;
  hecho.cerrar = ({ abortado = false, error = null } = {}) => {
    if (!m.enCurso) return;
    // El reparto del bucle, para que «va lento» tenga una causa y no cuatro
    // candidatas indistinguibles. Son acumulados de TODO el turno, prefill
    // incluido: el primer `forward` es el prompt entero.
    m.reparto = { forwardMs: msForward, readbackMs: msReadback, sampleMs: msSample, decodeMs: msDecode };
    m.totalMs = ahora() - tLlamada;
    m.enCurso = false;
    m.abortado = abortado;
    m.error = error;
    avisar(m);
  };

  const ctxLen = sesion.state.ctxLen;
  // Sitio reservado para contestar. Un prompt que ocupa el contexto ENTERO
  // «cabe» y luego revienta a mitad de la generación, con la respuesta ya
  // escrita a medias y perdida: más vale no empezar.
  // Razonando, la reserva es el presupuesto entero: si no, una conversación
  // larga dejaba 96 fichas para pensar Y responder.
  const razona = razonaEsteTurno();
  const topeModo = topeRespuesta(razona);
  const RESERVA = razona ? topeModo : 96;

  // El presupuesto de la historia sale del contexto REAL menos lo que ya ocupa
  // el sistema. Antes era un 4.000 fijo: en un modelo de 512 eso es pedirle al
  // gestor del historial que llene ocho veces lo que hay.
  const sistema = tk.encode(montarPrompt(system, [])).length;
  if (sistema + RESERVA > ctxLen) {
    throw new Error(`las instrucciones ocupan ${sistema} tokens y este modelo solo tiene ` +
      `${ctxLen} de contexto: elige un modelo con más contexto para trabajar con herramientas.`);
  }
  const mensajes = await ajustaHistorial(history, ctxLen - sistema - RESERVA);
  if (senal.aborted) throw errorParado();
  // El prompt se arma en DOS PIEZAS. `cabeza` es todo hasta el punto donde se
  // fotografía el estado; `cola` es lo que se alimenta después y queda FUERA de
  // la foto. Con modelo pensante la cola es el bloque `<think>`; sin él, el
  // último token, para que siempre quede algo con lo que sacar los logits.
  //
  // Se puede partir así porque está comprobado que el tokenizador NO fusiona en
  // la juntura: encode(cabeza + cola) === encode(cabeza) + encode(cola), 9 de 9
  // casos en los dos modelos. Si fusionara, el punto de corte no valdría.
  const pensar = pensante ? tk.encode(razona ? formato.abrir : formato.cerrado) : [];
  const armar = () => {
    const base = tk.encode(montarPrompt(system, mensajes, { conPensar: false }));
    return pensar.length
      ? { cabeza: base, cola: pensar }
      : { cabeza: base.slice(0, -1), cola: base.slice(-1) };
  };
  let { cabeza, cola } = armar();
  let ids = cabeza.concat(cola);
  // El gestor del historial cuenta en tokens ESTIMADOS (~4 chars/token) y este
  // tokenizador es el de verdad, así que puede pasarse. Se tiran turnos viejos
  // hasta que quepa, nunca el último: ese es lo que acaba de escribir el usuario.
  while (ids.length + RESERVA > ctxLen && mensajes.length > 1) {
    mensajes.shift();
    ({ cabeza, cola } = armar());
    ids = cabeza.concat(cola);
  }
  if (ids.length + RESERVA > ctxLen) {
    throw new Error(`tu mensaje ocupa más de lo que cabe en este modelo ` +
      `(${ids.length} tokens sobre ${ctxLen}): prueba con uno más corto.`);
  }
  hecho.prompt = ids.length;
  m.promptTokens = ids.length;

  // ── REUTILIZACIÓN DE PREFIJO ───────────────────────────────────────────────
  // Antes esto era `sesion.state.reset()` a secas, y el comentario lo justificaba
  // diciendo que el `system` cambia en cada llamada porque lleva el contexto vivo
  // dentro. Es cierto EN GENERAL y no lo es para los modelos lentos: los que
  // declaran `prefiereCompacto` reciben un prompt sin contexto vivo, estable
  // entre turnos. Reiniciando siempre, el turno 5 vuelve a pagar entero lo que ya
  // calcularon los turnos 1 a 4, y lo que se repaga crece con cada turno.
  //
  // La condición para reutilizar es la más estricta posible: que los tokens de
  // este turno EMPIECEN EXACTAMENTE por los que el estado ya ingirió. Entonces
  // basta alimentar la cola.
  //
  // Se compara sobre TOKENS y no sobre texto a propósito. Los tokens de la
  // respuesta que están en el estado son los que SE MUESTREARON; al rearmar el
  // prompt, el texto del asistente se vuelve a codificar, y destokenizar y
  // volver a tokenizar NO es la identidad. Comparando texto, esa deriva no se ve
  // y contamina en silencio todos los turnos siguientes.
  //
  // Y ante CUALQUIER divergencia se reinicia ENTERO, nunca a medias: la caché KV
  // se podría truncar en una posición, pero el estado recurrente del SSM —48 de
  // nuestras 64 capas— no se puede rebobinar, porque es el resultado de haber
  // recorrido toda la secuencia.
  // ENCENDIDO POR DEFECTO. Estuvo en opt-in hasta que la prueba pasó en LOS DOS
  // modelos, 6 asertos de 6 en cada uno: texto idéntico con y sin reutilización,
  // reutiliza 2 de 2 turnos posibles, y tocar un mensaje viejo fuerza reinicio
  // completo. El 27B se validó aparte del 0.8B a propósito, porque allí la
  // instantánea es unas ocho veces mayor (48 capas SSM, y más anchas).
  //
  // Se conserva la salida de emergencia —ponerlo a false— y no al revés: si esto
  // fallara no se vería como un error, se vería como una respuesta que habla de
  // otra conversación, y entonces hace falta poder apagarlo sin desplegar.
  const quiereReuso = globalThis.__ELFFUSS_PREFIJO !== false;
  // El motivo se calcula ANTES de tocar nada: cuando esto no reutilice hay que
  // poder saber POR QUÉ sin adivinar. «No reutiliza» tiene causas muy distintas
  // —primer turno, la historia se reordenó, el texto del asistente retokenizó
  // distinto, el prompt encogió— y desde fuera se ven todas iguales. La primera
  // versión no lo decía y el diagnóstico costó decodificar tokens a mano.
  //
  // La comparación es contra la CABEZA, que es lo que la instantánea describe.
  let motivo = 'primer turno';
  if (consumidos) {
    if (cabeza.length <= consumidos.length) motivo = `el prompt no crece (${cabeza.length} <= ${consumidos.length})`;
    else {
      const i = consumidos.findIndex((t, k) => t !== cabeza[k]);
      motivo = i === -1 ? 'ok' : `diverge en el token ${i} de ${consumidos.length}`;
    }
  }
  // ── DOS NIVELES DE RESCATE, no uno ────────────────────────────────────────
  // El nivel 1 es el de siempre: el prompt anterior completo es prefijo
  // estricto del nuevo, se restaura su foto y no se reprocesa nada. Solo se
  // cumple si la conversación crece por el final y NADA de en medio cambia.
  //
  // En un arnés eso casi nunca pasa: el estado vivo (archivo abierto, árbol,
  // tareas) cambia a cada paso, así que el nivel 1 falla y antes eso significaba
  // reprocesar el prompt ENTERO, prompt de sistema incluido. Medido: con un
  // prompt realista el prefill del modelo grande es la espera mas larga del
  // producto, y un
  // turno con herramientas lo pagaba en cada paso.
  //
  // El nivel 2 rescata lo que no cambia: si el bloque de sistema es el mismo
  // que describe la ranura 'sistema', se vuelve ahí y se reprocesa solo la
  // conversación. En Claw son ~1.300 fichas que se dejan de pagar; en Code,
  // ~2.200.
  const fichasSistema = tk.encode(bloqueSistema(system));
  // ⚠️ GUARDA, no suposición: que el bloque de sistema sea prefijo de fichas de
  // la cabeza no es gratis — el tokenizador podría fusionar en la juntura entre
  // el fin del bloque y el primer mensaje. Está comprobado para la juntura
  // cabeza/cola (9 de 9) pero NO para esta. Si no es prefijo exacto, el nivel 2
  // se apaga en vez de restaurar un punto que no es el que se cree.
  const sistemaEsPrefijo = fichasSistema.length < cabeza.length &&
    fichasSistema.every((t, k) => t === cabeza[k]);
  const mismoSistema = !!sistemaFoto && sistemaFoto.length === fichasSistema.length &&
    sistemaFoto.every((t, k) => t === fichasSistema[k]);

  let reusa = quiereReuso && motivo === 'ok' && sesion.state.tieneInstantanea('turno');
  if (reusa && !sesion.state.restaura('turno')) reusa = false;   // no se pudo: al camino seguro
  let yaIngeridos = reusa ? consumidos : null;
  let nivel = reusa ? 'turno' : null;

  if (!reusa && quiereReuso && sistemaEsPrefijo && mismoSistema &&
      sesion.state.tieneInstantanea('sistema') && sesion.state.restaura('sistema')) {
    reusa = true;
    yaIngeridos = fichasSistema;
    nivel = 'sistema';
    motivo += ' → rescatado por el prefijo del sistema';
  }
  // ── LO QUE EL TURNO VA A PISAR DEJA DE VALER, Y ANTES DE PISARLO ─────────
  // Una foto guarda el estado recurrente en un punto, pero la caché KV de las
  // posiciones anteriores a ese punto vive FUERA de la foto: restaurarla da por
  // hecho que nadie las ha reescrito. Un turno que reprocesa desde antes de una
  // foto las reescribe, y si se para —o falla— a medias, esa foto sigue
  // pareciendo válida sobre una caché que ya es de otra conversación.
  // Restaurarla después no da ningún error: da texto plausible sacado de una
  // mezcla de las dos.
  //
  // Así que lo que el turno va a pisar se invalida AQUÍ, antes del primer
  // forward, y cada foto se vuelve a apuntar solo cuando lo suyo está ingerido
  // entero: la del sistema tras su bloque, la del turno tras la cabeza.
  //   · reinicio: se reprocesa desde la posición 0, así que no sobrevive
  //     ninguna foto; `reset()` ya invalida la del turno, y la del sistema se
  //     invalida aquí y se vuelve a tomar en este mismo turno si se puede;
  //   · nivel 'sistema': se reprocesa desde el fin del bloque de sistema, así
  //     que la foto del TURNO, que es más larga, y `consumidos` dejan de valer;
  //   · nivel 'turno': se escribe detrás de las dos fotos y no se pisa nada.
  if (!reusa) {
    consumidos = null;
    sesion.state.reset();
    sistemaFoto = null;
    sesion.state.olvidaInstantanea('sistema');
  } else if (nivel === 'sistema') {
    consumidos = null;
    sesion.state.olvidaInstantanea('turno');
  }

  // ¿Hay que tomar la foto del sistema? Tras un reinicio no queda ninguna —se
  // acaba de invalidar—, así que se toma siempre que el bloque sea prefijo
  // exacto de lo que se va a procesar ahora.
  const tomarFotoSistema = quiereReuso && sistemaEsPrefijo && !reusa;
  const entradaCabeza = reusa ? cabeza.slice(yaIngeridos.length) : cabeza;
  // Lo PROCESADO de verdad son las dos fases: la cabeza que no se reutilizó más
  // la cola. Con la caché de prefijo funcionando esto es mucho menor que
  // `promptTokens`, y por eso se informan los dos: sin `reusados`, un prefill
  // que va «diez veces más rápido» parece un milagro o un error de medida.
  const procesados = entradaCabeza.length + cola.length;
  m.procesados = procesados;
  m.reusados = yaIngeridos ? yaIngeridos.length : 0;
  if (quiereReuso) {
    // El NIVEL va en la traza, y no es decoración: «reutilizados 24» puede ser
    // el prefijo completo de una conversación corta o el rescate del bloque de
    // sistema de una larga, y son dos cosas distintas que hay que poder
    // distinguir sin decodificar fichas a mano. Es la misma razón por la que el
    // motivo se calcula antes de tocar nada.
    console.log('[engine] prefijo: ' + (reusa
      ? `REUTILIZADOS ${yaIngeridos.length} tokens (nivel ${nivel}) · solo se procesan ${entradaCabeza.length + cola.length}`
      : `reinicio completo · ${ids.length} tokens · ${motivo}`));
  }

  const paradas = new Set([tk.eosId]);
  const imEnd = tk.encode('<|im_end|>');
  if (imEnd.length === 1) paradas.add(imEnd[0]);

  const vocab = tk.vocabSize;
  const salida = [];
  hecho.salida = salida;
  let texto = '';
  const decS = new TextDecoder('utf-8');
  // Un tokenizador sin `bytesDe` (los de las pruebas) da sus bytes por decode.
  const utf8 = new TextEncoder();
  const bytesDe = tk.bytesDe ? (id) => tk.bytesDe(id) : (id) => utf8.encode(tk.decode([id]));
  // El reloj del prefill arranca AQUÍ, antes de la fase 1, y no en el bucle.
  //
  // Esto estaba mal y lo cazó una prueba interna: el prompt se parte en
  // CABEZA (lo que entra en la instantánea, la inmensa mayoría) y COLA, y la
  // cabeza se procesa aquí, fuera del bucle. Midiendo solo desde el bucle, un
  // prompt de ~780 tokens declaraba «4 tokens procesados», y la velocidad que
  // salía de ahí no significaba nada: el tiempo real de los otros ~776 se
  // escondía en el hueco entre la llamada y el bucle. Se veía porque el tiempo
  // hasta la primera letra era decenas de veces el «prefill».
  const tPrefillIni = ahora();
  // ── FASE 1: alimentar hasta el punto de la foto ────────────────────────────
  // Con la señal dentro, y en los DOS tramos cuando se parte para fotografiar el
  // sistema: un prompt largo son muchos lotes y muchas capas, y parar tiene que
  // poder entrar entre ellos en vez de esperar al final.
  hecho.fase = 'prefill';
  if (entradaCabeza.length) {
    // El progreso, solo si alguien lo escucha; `desde` es lo ya hecho en un
    // tramo anterior, porque se reporta sobre el TOTAL de la cabeza. Y nada
    // de progreso con la señal ya dada: `forward` avisa de cada lote
    // terminado, también del que acaba justo cuando llega el «para» y del
    // último, que no mira la señal; pero el contrato de `chat()` es que tras
    // notar el «para» no se llama más a `onPrefill`, así que se filtra aquí.
    const opciones = (desde, total) => (onPrefill
      ? { signal: senal, onProgress: (h) => { if (!senal.aborted) onPrefill(desde + h, total); } }
      : { signal: senal });
    if (tomarFotoSistema) {
      // Se parte el forward de la cabeza para poder fotografiar EXACTAMENTE
      // después del bloque de sistema. Partirlo no cambia la numérica: el
      // estado avanza igual token a token (lo vigila una prueba interna que
      // compara troceado contra una sola pasada).
      const n = fichasSistema.length;
      // El progreso se reporta sobre el TOTAL de la cabeza, no sobre cada
      // tramo: al usuario le da igual que por dentro se parta para fotografiar.
      const total = entradaCabeza.length;
      await sesion.forward(entradaCabeza.slice(0, n), opciones(0, total));
      sesion.recycle();
      sesion.state.instantanea('sistema');
      sistemaFoto = fichasSistema;
      if (entradaCabeza.length > n) {
        await sesion.forward(entradaCabeza.slice(n), opciones(n, total));
        sesion.recycle();
      }
    } else {
      await sesion.forward(entradaCabeza, opciones(0, entradaCabeza.length));
      sesion.recycle();
    }
  }
  // La foto se toma AQUÍ, antes del bloque `<think>`. Lo que describe —la
  // cabeza— SÍ es prefijo exacto del prompt del turno siguiente, y eso es lo
  // que hacía imposible la reutilización mientras el bloque estaba dentro.
  //
  // Se apunta ahora y no al final del turno a propósito: si la generación
  // revienta a media respuesta, la instantánea sigue describiendo un punto
  // anterior y VÁLIDO, así que el turno siguiente puede reutilizar igual. Antes
  // había que invalidar por si acaso.
  sesion.state.instantanea('turno');
  consumidos = cabeza;
  // Se mira DESPUÉS de la foto y de `consumidos`, no antes: la cabeza ya está
  // ingerida entera, y apuntarlo deja que el turno siguiente la reutilice
  // aunque este se pare aquí mismo.
  if (senal.aborted) throw errorParado();
  hecho.fase = 'generación';

  // ── FASE 2: la cola, que entra en el estado pero NO en la foto ─────────────
  let entrada = cola;

  // `maxTokens` por muestreo: el jurado de rlm.js y los tests necesitan acotar
  // la respuesta, y con un modelo lento la diferencia entre 8 y 512 tokens es la
  // diferencia entre una prueba que se corre y una que no.
  //
  // El tope de verdad, además, es lo que QUEDA de contexto: con 512 de contexto
  // y 512 de tope, una respuesta larga se llevaba por delante la caché a mitad
  // de escribirla y el turno entero se perdía como error.
  const tope = Math.max(1, Math.min(topeModo, ctxLen - ids.length));
  // Traza de cada turno. No es depuración de usar y tirar: cuando este modelo
  // se queda mudo —y se queda— lo ÚNICO que distingue «no le cabía», «le
  // pusieron un tope de 0» y «decidió callarse» son estos tres números, y sin
  // ellos hay que adivinar. Cuesta una línea de consola por respuesta.
  console.log(`[engine] turno: prompt ${ids.length} tokens · contexto ${ctxLen} · ` +
    `tope ${tope}${razona ? ' (razonando)' : ''} · muestreo ${muestreo ? JSON.stringify(muestreo) : muestreoDelRegistro() ? 'del registro ' + JSON.stringify(muestreoDelRegistro()) : 'voraz (por defecto)'}`);
  // `tLlamada` guarda la entrada a `chat()` para poder dar el tiempo hasta la
  // primera letra TAL COMO LO ESPERA UN HUMANO, que incluye la tokenización y el
  // montaje del prompt.
  let tPrefill = 0;
  // ── DÓNDE SE VA EL TIEMPO DE CADA TOKEN ────────────────────────────────────
  // La generación medida de punta a punta salió MUY por debajo de lo que da el
  // kernel aislado para el mismo modelo, y el bucle tiene tres sospechosos que
  // la medida del kernel suelto no paga nunca:
  //
  //   · el readback trae el vocabulario ENTERO de vuelta a CPU en cada token;
  //   · `sample` recorre todos esos logits, y con penalización de repetición
  //     además recorre todo lo emitido hasta ahora;
  //   · `tk.decode(salida)` redecodifica la secuencia COMPLETA en cada token,
  //     que es cuadrático en la longitud de la respuesta.
  //
  // Sin repartirlo, «la generación va lenta» tiene cuatro causas que se ven
  // idénticas desde fuera. Cuesta tres relojes por token. Los acumuladores
  // nacen con la medida, arriba, para que un turno parado también los cierre.
  for (let paso = 0; paso < tope; paso++) {
    // Parar se mira en DOS sitios por paso: aquí, antes de encolar trabajo, y
    // tras leer los logits, antes de emitir. El segundo es el que garantiza que
    // después de notar el «para» no sale ni un token más: la lectura es la
    // espera larga del paso, y es justo ahí donde suele llegar.
    if (senal.aborted) throw errorParado();
    const tF = ahora();
    const { logits, logitsTokens } = await sesion.forward(entrada, { signal: senal });
    // El forward solo devuelve logits del ÚLTIMO token: antes calculaba los
    // 248.320 de CADA token del prompt —la cabeza entera, una vez por token— y
    // se descartaban todos menos los del último.
    const tR = ahora(); msForward += tR - tF;
    const todos = new Float32Array(await readback(gpu.device, logits, 0, logitsTokens * vocab * 4));
    if (senal.aborted) throw errorParado();
    const ultimo = todos.subarray((logitsTokens - 1) * vocab, logitsTokens * vocab);
    const tS = ahora(); msReadback += tS - tR;
    // Con los tokens ya emitidos, para que la penalización de repetición pueda
    // actuar. Sin ellos, el muestreador no sabe que se está repitiendo.
    // La semilla CAMBIA en cada token (como en generate()): con la misma en
    // todos, cada paso sacaba el mismo cuantil de su distribución y no había azar.
    const mu = muestreoEfectivo() || { temperature: 0 };   // (`m` es la medida del turno)
    const id = sample(ultimo, { ...mu, seed: (mu.seed || 1) + paso,
                                previos: ids.concat(salida) });
    msSample += ahora() - tS;
    sesion.recycle();

    // Frontera prefill/generación: el primer paso ha metido `procesados` tokens
    // de golpe y ya hay logits leídos, así que aquí acaba el prefill. Se marca
    // DESPUÉS del readback a propósito: medir antes daría un prefill
    // fantásticamente rápido y un primer token absurdamente lento, porque el
    // trabajo del dispositivo se habría contado en el token siguiente.
    if (paso === 0) {
      tPrefill = ahora();
      m.prefillMs = tPrefill - tPrefillIni;
      m.prefillTokS = m.prefillMs > 0 ? procesados / (m.prefillMs / 1000) : 0;
      avisar(m);
    }

    if (paradas.has(id)) break;
    salida.push(id);
    // La generación se cuenta desde el FIN del prefill, no desde el principio:
    // si no, el primer token arrastra el prompt entero y los tok/s de
    // generación de una respuesta corta serían los del prefill disfrazados.
    m.genTokens = salida.length;
    m.genMs = ahora() - tPrefill;
    m.genTokS = m.genMs > 0 ? m.genTokens / (m.genMs / 1000) : 0;
    if (!m.ttftMs) m.ttftMs = ahora() - tLlamada;
    avisar(m);

    // Se decodifica la secuencia ENTERA y se emite la diferencia: un token puede
    // partir un carácter multibyte por la mitad, y decodificarlo suelto daría un
    // rombo de reemplazo en mitad de una palabra.
    // EN STREAMING sobre los bytes de cada token, no redecodificando la salida
    // entera en cada paso (cuadrático: con miles de tokens de razonamiento se
    // notaba en la velocidad). El decodificador guarda los bytes de un carácter
    // a medias hasta que llega el resto, así que tampoco emite rombos de
    // reemplazo que luego hay que corregir. El texto final es el mismo que el
    // de decodificarlo todo de una vez: así lo define el estándar.
    const tD = ahora();
    const trozo = decS.decode(bytesDe(id), { stream: true });
    if (trozo) {
      onToken(trozo);
      texto += trozo;
      hecho.texto = texto;
    }
    msDecode += ahora() - tD;
    entrada = [id];
  }
  { const fin = decS.decode(); if (fin) { onToken(fin); texto += fin; hecho.texto = texto; } }

  // El razonamiento se transmite EN CRUDO por onToken —la burbuja de
  // pensamiento de Elffuss está hecha para leerse— pero se limpia del texto que
  // se devuelve: el agente busca ahí bloques ```tool y no debe ver las marcas.
  // Mismo tratamiento que providers/onnx.js, para no inventar otra convención.
  const limpio = razona ? limpiarRazonando(texto) : limpiarRazonamiento(texto);
  if (razona && !texto.includes('</think>')) {
    console.log(`[engine] razonamiento CORTADO por el tope (${tope} tokens) sin llegar a «</think>»: ` +
      'la respuesta queda vacía, no se devuelve el razonamiento como si fuera la respuesta');
  }
  // Cierra la traza del turno con lo que SALIÓ. Sin esto, «no aparece burbuja»
  // tiene dos causas que se ven idénticas desde fuera y llevan a sitios
  // opuestos: que el modelo no emitiera nada, o que emitiera algo que la capa de
  // arriba se comió —un bloque ```tool mal formado, por ejemplo, que el agente
  // consume sin pintar nada—. Distinguirlas es la diferencia entre culpar al
  // motor y culpar al agente, y nos ha costado una tarde entera no poder.
  console.log(`[engine] turno: salieron ${salida.length} tokens · ` +
    `crudo ${texto.length} car. · devuelto ${limpio.length} car.` +
    (texto ? ` · empieza por ${JSON.stringify(texto.slice(0, 80))}` : ' · NADA'));
  hecho.cerrar();
  console.log(`[engine] velocidad: prefill ${m.procesados} tok en ` +
    `${m.prefillMs.toFixed(0)} ms (${m.prefillTokS.toFixed(1)} tok/s` +
    `${m.reusados ? `, ${m.reusados} reusados del prefijo` : ''}) · ` +
    `generación ${m.genTokens} tok a ${m.genTokS.toFixed(1)} tok/s · ` +
    `primera letra ${m.ttftMs.toFixed(0)} ms`);
  return limpio;
}

// Razonando, la apertura «<think>» va en el PROMPT: la salida trae el
// razonamiento, el «</think>» y la respuesta. Lo que va después del último
// cierre es la respuesta. Y si el tope cortó antes de cerrar, NO hay respuesta:
// devolver el razonamiento a medias como respuesta haría que el agente buscase
// herramientas en lo que el modelo solo estaba pensando.
function limpiarRazonando(txt) {
  const i = txt.lastIndexOf('</think>');
  if (i === -1) return '';
  return txt.slice(i + 8).replace(/<\/?think>/g, '').trim();
}

function limpiarRazonamiento(txt) {
  if (txt.includes('<think>')) {
    txt = txt.replace(/<think>[\s\S]*?<\/think>/g, '');
    // Si el tope de tokens cortó el bloque a medias, nos quedamos con lo que
    // vino DESPUÉS de la última apertura: devolver un razonamiento sin cerrar
    // como si fuera la respuesta es peor que devolver poco.
    const i = txt.lastIndexOf('<think>');
    if (i !== -1) txt = txt.slice(i + 7);
  } else if (txt.includes('</think>')) {
    // Cierre SUELTO, sin apertura. Pasa cuando la apertura va en el PROMPT —que
    // es como lo hace la plantilla de estos modelos— y el modelo solo emite el
    // cierre. La rama de arriba no lo veía y el razonamiento entero se colaba en
    // la respuesta, donde el agente busca bloques ```tool.
    txt = txt.slice(txt.lastIndexOf('</think>') + 8);
  }
  return txt.replace(/<\/?think>/g, '').trim();
}

// Gancho de pruebas: el respaldo mixto (almacén + red) se verifica con una
// prueba que necesita llamar a esto directamente.
export { rangosRepartidos as __rangosRepartidos };

// Para las pruebas: el prompt que se montaría con la plantilla dada, sin cargar
// ningún modelo. Deja el formato y `pensante` como estaban.
export function __montarConPlantilla(plantilla, tok, system, mensajes, { razona = false, arch = 'lfm3' } = {}) {
  const antes = [formato, pensante];
  try {
    formato = formatoDePlantilla(plantilla, tok, arch);
    pensante = /<think>/.test(String(plantilla || ''));
    return montarPrompt(system, mensajes, { conPensar: false }) + (pensante ? (razona ? formato.abrir : formato.cerrado) : '');
  } finally { [formato, pensante] = antes; }
}
