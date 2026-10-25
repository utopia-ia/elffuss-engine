// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · el REGISTRO de modelos: qué hay, dónde vive y cómo se lee.
// ─────────────────────────────────────────────────────────────────────────────
// Vive aparte del proveedor por una razón concreta: la app necesita SONDEAR si
// un modelo está disponible antes de ofrecerlo en el selector, y hacerlo
// importando el proveedor entero costaría traerse el motor completo a todo el
// que abra la página sin usarlo nunca.
//
// Tener las URLs en dos sitios ya salió mal una vez: cuando el 27B pasó de ser
// un fichero suelto a dos fragmentos en dos hosts, el sondeo de la app se quedó
// apuntando a la URL VIEJA, y nada lo habría dicho hasta que esa URL dejara de
// responder. Una sola fuente.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Registro de modelos. `bytes` no es decorativo: se compara con el
// content-length real y avisa si el fichero de origen ha cambiado, porque un
// cambio de cuantización aguas arriba altera la salida sin previo aviso.
export const MODELS = {
  // MiniCPM5-2B en Q8_0.
  //
  // Entra sin tocar el motor porque su GGUF declara `general.architecture =
  // llama`, que ya se lee. Es un transformer PURO: las 42 capas son de atención
  // y las 42 pagan contexto, así que no tiene la ventaja del híbrido —en los
  // Qwen la mayoría de los bloques son SSM con estado de tamaño fijo— y su
  // caché KV por ficha es casi la del 27B teniendo diez veces menos parámetros.
  //
  // El fichero declara un contexto de 131.072 y aquí se pide 65.536. Lo que
  // manda no es el modelo, es lo que cuesta su KV: a 131k en f16 no cabe en el
  // presupuesto de una pestaña, y a 65k sí. Con la KV a 8 bits —no implementada
  // todavía, hoy solo hay f16 y f32— el contexto declarado sería alcanzable.
  //
  // Q8_0 y no algo más pequeño a propósito: es uno de los dos únicos tipos que
  // entran por el GEMM teselado del prefill. Un Q4_K cae al camino por filas y
  // paga una relectura de los pesos POR CADA ficha del lote.
  'minicpm5-2b': {
    label: 'MiniCPM5 2B · Q8_0',
    url: 'https://m1.elffuss.com/minicpm5-2b.q8',
    bytes: 2679710688,
    ctx: 65536,
    // RAZONA, con presupuesto amplio. En tareas de agente, con el razonamiento
    // cerrado se paraba al segundo paso sin tocar ningún fichero, y razonando sí
    // avanzaba; darle más presupuesto que este no lo mejoraba. provider.js abre
    // el bloque y cuenta el razonamiento dentro de este tope.
    razona: true,
    presupuesto: 4000,
  },

  'qwen35-0.8b': {
    label: 'Qwen3.5 0.8B · Q8_0',
    url: 'https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf',
    bytes: 833592096,
    // 64k, medido en los dos modelos: la reserva funciona y el prefill NO cambia
    // (el contexto cuesta memoria, no tiempo — solo se paga por los tokens que
    // de verdad se procesan). Con la caché en f16 son la mitad de bytes que la
    // cuenta de antes: el modelo grande pide la mitad de memoria para la misma
    // ventana.
    //
    // Si no cupiera, `provider.js` baja por mitades hasta 2048 en vez de dejar
    // al usuario sin modelo: un contexto de 16k sigue siendo ocho veces el de
    // antes y no se parece en nada a no tener modelo.
    ctx: 65536,
  },
  // El 27B, en dos fragmentos. Es una opción avanzada por lo que CUESTA, no por
  // lo que falle: la descarga es grande y el prefill de un prompt largo, lento;
  // el retraso cae entero antes del primer token. llama.cpp no lee su formato,
  // así que se comprueba internamente por piezas —el formato contra su
  // definición publicada, la rotación contra la matriz explícita, que se aplique
  // a todos los tensores que la declaran— y de extremo a extremo: que conteste
  // con sentido.
  //
  // El `bytes` es el del fichero verificado: si el de origen cambia de
  // cuantización, la salida cambia sin avisar y esto lo caza.
  'qwen38-27b': {
    // El id es el de antes por compatibilidad (las apps lo escriben a mano); el
    // modelo es el ternario de Prism ML. Frente a las alternativas del mismo
    // tamaño, su matmul gana porque NO hay codebook: el peso sale de un
    // desplazamiento y una máscara, no de una tabla de rejilla con gather
    // disperso por carril.
    //
    // Se elige el formato PQ2_0 y no el PTQ1_0 aunque ocupe más: 128 pesos a
    // 2 bits son 32 bytes JUSTOS y, con subgrupos de 32 carriles, toca un byte
    // por carril, sin divergencia. El de base 3 reparte en tres etapas de tamaños
    // distintos y va 1,53x más lento. Y se puede trasladar la medida de calidad
    // entre los dos porque guardan los MISMOS pesos bit a bit (comprobado en 4
    // tensores, 20.480 valores).
    //
    // ⚠️ Requiere la rotación de Hadamard sobre las activaciones: el fichero la
    // declara en sus metadatos y `hadamard.js` la aplica. Un motor que no la
    // tenga NO falla — carga y devuelve ruido.
    label: 'Bonsai 27B · ternario',
    // Servido desde el host del BROKER, no desde la web. Ahí el broker lo lee
    // en su MISMO origen (sin CORS de por medio) y lo guarda en el almacén de
    // ese origen, pensado para que las webs de Elffuss no se lo bajen cada una.
    // Y el host deja leerlo por CORS a las webs de Elffuss, así que desde ellas
    // el camino directo —sin broker— también funciona.
    //
    // Servido desde la propia web era IMPOSIBLE compartirlo: el broker vive en
    // otro origen y sin CORS su fetch fallaba con «Failed to fetch», sin más
    // explicación.
    // REPARTIDO entre dos hosts, y la razón es medible: la cuota del navegador
    // es POR ORIGEN. En uno solo el modelo entero no cabía fiable; en dos, cada
    // mitad va sobrada. Y como los dos son subdominios del mismo sitio, lo
    // guardado puede servir a las demás webs de Elffuss sin volver a bajarlo.
    //
    // `bytes` por fragmento NO es decorativo: si uno cambia de tamaño, las
    // fronteras se desplazan y todo lo que venga detrás se lee corrido — basura
    // sin un solo error.
    url: [
      { url: 'https://m1.elffuss.com/bonsai-27b.p0', bytes: 3602907136 },
      { url: 'https://m2.elffuss.com/bonsai-27b.p1', bytes: 3603261792 },
    ],
    // El tamaño exacto del fichero servido (los dos fragmentos juntos): si
    // cambia, el motor avisa en cada carga de que el origen ha cambiado.
    bytes: 7206168928,
    // 512 era INVIABLE y no por poco: el prompt de sistema de Elffuss son 626
    // tokens él solo (medido con este mismo tokenizador), así que la caché KV
    // reventaba —«la caché se llena»— antes de que el usuario escribiera nada.
    // El modelo no fallaba: no llegaba a ver la pregunta.
    //
    // Subirlo NO cuesta tiempo, cuesta memoria: `ctx` solo dimensiona la caché
    // KV (cuatro veces más a 2048 que a 512). El prefill lo paga el número de
    // tokens que se procesan de verdad, que es el mismo. Bajarlo para «ir más
    // rápido» fue confundir las dos cosas.
    ctx: 65536,   // ver la nota del modelo pequeño: medido, y con red si no cabe
    // SE CACHEA: el almacén del anfitrión lo guarda entre visitas, y la segunda
    // carga no vuelve a pedirlo por red.
    cache: true,
    // El muestreo de la ficha de Prism para responder SIN razonar (que es como
    // lo usa el motor): 0,7 / 0,8 / top-k 20 / min-p 0 / presence 1,5, y sin la
    // penalización de repetición: es lo que recomienda quien lo cuantizó. Y
    // respuestas de hasta 2000 fichas: escribe ficheros enteros, y 512 los
    // cortaba.
    muestreo: { temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presence: 1.5, repeatPenalty: 1 },
    tope: 2000,
    // El prompt compacto ESTUVO aquí y se ha quitado. La razón de ponerlo era
    // de velocidad: el completo tarda más en procesarse y el usuario espera
    // mirando una caja quieta. Pero un prompt recortado deja al modelo sin
    // saber hacer tareas, y un asistente que carga, responde y no sirve para
    // nada es peor que uno lento — probado en uso real, no en un banco.
    //
    // El coste de volver al prompt completo es un primer prefill más largo.
    // Eso se ataca donde toca: reutilizando el prefijo entre turnos (ya hecho,
    // ver provider.js) y guardando la foto del estado entre SESIONES, que es lo
    // que hace que se pague una vez y no cada vez que se abre la web.
  },
};

// ¿Se puede usar este modelo AHORA? Se pregunta a los ficheros de verdad, con
// una petición de un solo byte por fragmento: comprueba de paso que el host
// responda, que admita rangos y que el CORS deje leerlo, que es exactamente lo
// que hará la carga después. Ofrecer una opción que falla al elegirla es peor
// que no ofrecerla.
// Todos los ORIGENES donde puede haber pesos guardados.
//
// Cada fragmento lo guarda el broker de SU PROPIO host (ver `model-broker.js`),
// asi que un modelo repartido entre dos servidores ocupa DOS origenes. El panel
// de almacenamiento preguntaba solo al broker por defecto y por eso enseñaba
// una fraccion: con el 27B guardado en m1 y m2, el contador miraba a un tercer
// origen que no tiene nada. Una cifra de ocupacion que no suma donde estan los
// bytes es peor que no tenerla, porque parece exacta.
export function origenesAlmacen() {
  // Solo los origenes NUESTROS: un modelo servido desde un tercero (Hugging
  // Face, por ejemplo) no tiene nuestra pagina de broker, asi que preguntarle
  // no devolveria nada y ademas costaria la espera entera del timeout, con el
  // panel colgado por un origen que nunca iba a responder.
  const NUESTRO = /(^|\.)elffuss\.com$|(^|\.)utopiaia\.com$|^localhost$/;
  const o = new Set();
  for (const m of Object.values(MODELS)) {
    const urls = Array.isArray(m.url) ? m.url.map(f => f.url) : (m.url ? [m.url] : []);
    for (const u of urls) {
      try {
        const { hostname, origin } = new URL(u);
        if (NUESTRO.test(hostname)) o.add(origin + '/');
      } catch { /* url relativa o rara */ }
    }
  }
  return [...o];
}

export async function disponible(id) {
  const m = MODELS[id];
  if (!m) return false;
  const urls = Array.isArray(m.url) ? m.url.map(f => f.url) : [m.url];
  try {
    const rs = await Promise.all(urls.map(u =>
      fetch(u, { headers: { Range: 'bytes=0-0' }, cache: 'no-store' })));
    return rs.every(r => r.ok);
  } catch { return false; }
}

// ¿Se guarda este modelo en el navegador? Normalmente lo dice el registro, pero
// se puede FORZAR desde fuera con `localStorage['elffuss.engine.cache']`
// ('1' enciende, '0' apaga).
//
// No es un ajuste para usuarios, es un instrumento: el cacheo del 27B solo se
// comprueba de verdad en una web servida, y sin este interruptor la única forma
// de probarlo sería encendérselo a todo el mundo. Si sale mal, quien lo paga es
// quien esté cargando el modelo en ese momento: la descarga entera perdida y
// vuelta a empezar por el otro camino.
export function cachea(m) {
  try {
    const o = localStorage.getItem('elffuss.engine.cache');
    if (o === '1') return true;
    if (o === '0') return false;
  } catch { /* sin localStorage (worker, sandbox) */ }
  return m.cache !== false;
}
