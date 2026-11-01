// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · atención FUSIONADA: puntuaciones, softmax y combinación en
// un solo despacho, sin escribir nada intermedio en memoria global.
// ─────────────────────────────────────────────────────────────────────────────
// La cadena de siempre son cuatro despachos más la puerta (`kernels.attn.wgsl.js`
// y el softmax de `kernels.nn.wgsl.js`): las puntuaciones [H][T][nKv] van a
// memoria global, el softmax las relee tres veces, la combinación parte el
// contexto en trozos y una reducción los junta. Y la máscara causal (y la de
// ventana) se aplica SOLO en el softmax, así que las puntuaciones y la
// combinación calculan también las claves que la máscara tira a la basura. En
// un prefill eso es la mitad del trabajo de la atención en las capas globales y
// bastante más en las de ventana.
//
// Este kernel hace lo mismo con el softmax EN LÍNEA (máximo y suma corrientes,
// en f32): cada grupo recorre sus claves en teselas de 32, reescala lo
// acumulado cuando sube el máximo y no guarda ninguna puntuación. Las teselas
// que caen enteras fuera de la máscara —más allá de la causal o antes del
// principio de la ventana— ni se visitan.
//
// Reparto: un grupo por (cabeza KV, tesela de TQ consultas, parte del rango de
// claves), con las G = H/KVH cabezas de consulta de esa cabeza KV DENTRO del
// grupo. Así cada tesela de K se lee una vez y la usan G·TQ filas de consulta.
//   · Q·Kᵀ: un carril por CLAVE. Cada carril hace el producto de su clave con
//     las RS filas de su subgrupo; q se lee en difusión (todos los carriles la
//     misma dirección) y la fila de K de cada carril sale de memoria de grupo.
//   · softmax: el máximo y la suma de la tesela son un `subgroupMax` y un
//     `subgroupAdd` por fila: los 32 carriles son las 32 claves.
//   · P·V: un carril por DIMENSIONES (D/32 cada uno). La probabilidad de la
//     clave j la reparte `subgroupShuffle` desde el carril j, y la fila de V se
//     lee directamente de la caché: 32 carriles leen un tramo contiguo.
//     OJO: así cada SUBGRUPO lee la tesela de V entera, y con NSG subgrupos la
//     misma tesela se pide NSG veces; la que se lee una sola vez por grupo es
//     la K. Con `vsh` la V también pasa una vez por memoria de grupo (a costa
//     de otra tesela de memoria y de una copia); cuál de las dos gana lo decide
//     el barrido, ver `ops.flash.js`.
//
// La K pasa por memoria de grupo porque hay que TRASPONERLA: se lee de la
// caché con los carriles a lo largo de la fila (lectura contigua) y se usa con
// un carril por fila. Las filas van con los trozos de 16 bytes permutados en XOR
// con la fila: sin eso los 32 carriles leerían el mismo banco a la vez (las
// filas miden un múltiplo exacto del ancho de los bancos) y la lectura se
// serializaría entera.
//
// La memoria de grupo se mantiene en el presupuesto que fija `ops.flash.js` a
// propósito: un grupo que se come casi todo el tope deja UN grupo residente por
// unidad de cómputo y no queda con qué esconder latencias (ya pasó con el
// primer scores con K compartida, ver `kernels.attn.wgsl.js`).
//
// Supuestos que comprueba `ops.flash.js` antes de elegir este camino:
//   · subgrupos de 32 carriles exactos, y un grupo de 32·NSG hilos formado por
//     NSG subgrupos CONSECUTIVOS (los 32 primeros hilos son el primer subgrupo,
//     los 32 siguientes el segundo, y así): el mismo supuesto que ya usa el
//     matmul con varios subgrupos;
//   · caché KV en f16, leída como palabras de 32 bits y abierta en sus dos
//     mitades con `unpack2x16float` (del núcleo de WGSL: no hace falta
//     `enable f16`);
//   · D múltiplo de 128 (D/32 dimensiones por carril en vec4) y D/8 potencia de
//     dos (la permutación XOR).
//
// NO es bit a bit igual a la cadena de siempre: el producto Q·K se suma en
// otro orden (vec4 en vez de escalar), el softmax reescala por teselas y la
// combinación suma las claves en otro orden. Mismas claves visibles, misma
// escala, misma puerta: la diferencia es de redondeo, y se valida contra la
// cadena de siempre con una tolerancia declarada.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Claves por tesela: una por carril de subgrupo.
export const FLASH_BK = 32;

// Bytes de memoria de grupo que pide el kernel. Vive aquí, junto al shader que
// declara los arrays, para que `ops.flash.js` no pueda desincronizarse: si las
// dos cuentas discreparan, la validación del pipeline fallaría en el mejor caso
// y en el peor se pediría más de lo previsto sin enterarse.
export function flashMemoria({ D, G, TQ, qsh = true, vsh = false, psh = false, NSG = 1 }) {
  const kBytes = FLASH_BK * D * 2;               // tesela de K en f16
  const qBytes = qsh ? G * TQ * D * 4 : 0;       // filas de q en f32
  const vBytes = vsh ? FLASH_BK * D * 2 : 0;     // tesela de V en f16
  // Probabilidades de la tesela: por subgrupo y clave, sus RS filas en vec4.
  const pBytes = psh ? NSG * FLASH_BK * Math.ceil(G * TQ / NSG / 4) * 16 : 0;
  return kBytes + qBytes + vBytes + pBytes;
}

// Registros vec4<u32> por hilo que necesita la prelectura de K (la tesela
// repartida entre los hilos del grupo), o 0 si no se puede: el reparto tiene
// que ser exacto y caber en pocos registros, porque viven durante todo el
// cálculo de la tesela.
export function flashPrefetch({ D, NSG }) {
  const total = FLASH_BK * (D / 8), WG = 32 * NSG;
  const nr = total % WG === 0 ? total / WG : 0;
  return nr > 0 && nr <= 8 ? nr : 0;
}

function comprueba({ D, G, TQ, NSG }) {
  if (![128, 256].includes(D)) throw new Error(`flash: headDim ${D} sin soporte (128 o 256)`);
  if (!(G >= 1 && TQ >= 1 && NSG >= 1) || [G, TQ, NSG].some((x) => !Number.isInteger(x))) {
    throw new Error(`flash: forma inválida G=${G} TQ=${TQ} NSG=${NSG}`);
  }
  if ((G * TQ) % NSG !== 0) throw new Error(`flash: ${G * TQ} filas no se reparten entre ${NSG} subgrupos`);
}

// El kernel. Parámetros de COMPILACIÓN (todos van en la clave del pipeline):
//   D        dimensión de cabeza
//   G        cabezas de consulta por cabeza KV
//   TQ       consultas (tokens) por grupo; 1 en decode
//   NSG      subgrupos por grupo: RS = G·TQ/NSG filas por subgrupo
//   puerta   epílogo × sigmoide(puerta[t][h]) (lfm3); solo sin `parcial`
//   parcial  escribe sumas parciales (o, m, l) por parte del rango de claves y
//            deja la normalización y la puerta a `flashReduceShader`
//   qsh      filas de q en memoria de grupo (si no, lectura en difusión de global)
//   vsh      tesela de V también en memoria de grupo (si no, de la caché directa)
//   pref     la tesela de K SIGUIENTE se pide a registros antes de calcular la
//            actual, y se vuelca a memoria de grupo en la vuelta siguiente: la
//            espera de la lectura queda detrás del cálculo en vez de delante.
//            Mismos datos y mismos índices; cuesta registros (ver flashPrefetch).
//   psh      las probabilidades llegan a P·V por memoria de grupo y no por
//            `subgroupShuffle`: cada carril deja las RS de su clave en vec4 y
//            P·V lee, por clave, un vec4 en difusión que sirve a cuatro filas,
//            en vez de RS barajados. A cambio, una barrera más por tesela.
//            Mismos valores y mismo orden de suma: cambia solo el camino.
export function flashShader({ D, G, TQ, NSG, puerta = false, parcial = false, qsh = true, vsh = false, pref = false, psh = false }) {
  comprueba({ D, G, TQ, NSG });
  const R = G * TQ, RS = R / NSG, WG = 32 * NSG;
  const PV = Math.ceil(RS / 4);   // vec4 de probabilidades por clave (psh)
  const NR = flashPrefetch({ D, NSG });
  if (pref && !NR) throw new Error(`flash: sin prelectura para D=${D} con ${NSG} subgrupos`);
  const CPR = D / 8;          // trozos de 8 mitades (un vec4<u32>) por fila de K
  const SWZ = CPR - 1;        // máscara de la permutación XOR de trozos
  const QV = D / 4;           // vec4<f32> por fila de q y de salida
  const DPL = D / 32;         // dimensiones por carril en P·V
  const NV4 = DPL / 4;        // vec4<f32> por carril
  const VT = DPL === 4 ? 'vec2<u32>' : 'vec4<u32>';   // DPL mitades de una fila de V
  const conPuerta = puerta && !parcial;
  const F = Array.from({ length: RS }, (_, i) => i);
  const E = Array.from({ length: NV4 }, (_, e) => e);

  let b = 0;
  const bQ = b++, bK = b++, bV = b++;
  const bG = conPuerta ? b++ : -1;
  const bOut = parcial ? -1 : b++;
  const bPO = parcial ? b++ : -1;
  const bPM = parcial ? b++ : -1;
  const bP = b++;

  // P·V de U en U claves: las U lecturas de V van primero y son independientes,
  // así que pueden estar todas en vuelo a la vez en vez de esperar una por una.
  const U = 4;
  const Us = Array.from({ length: U }, (_, u) => u);
  // Las DPL mitades de la fila de V del carril, abiertas en NV4 vec4<f32>.
  const vvDe = (u) => (NV4 === 1
    ? `      let vv${u}_0 = vec4<f32>(unpack2x16float(vw${u}.x), unpack2x16float(vw${u}.y));`
    : `      let vv${u}_0 = vec4<f32>(unpack2x16float(vw${u}.x), unpack2x16float(vw${u}.y));
      let vv${u}_1 = vec4<f32>(unpack2x16float(vw${u}.z), unpack2x16float(vw${u}.w));`);
  const vLee = (u) => (vsh
    ? `      let vw${u} = vs[(j0 + ${u}u) * 32u + lane];`
    : `      let vw${u} = v[(min(kb + j0 + ${u}u, pFin - 1u) * p.KVH + kvh) * 32u + lane];`);

  const qLee = (i, off) => (qsh ? `qs[qb${i} + ${off}]` : `q[qb${i} + ${off}]`);
  // psh: las PV palabras vec4 de la clave j0 + u en este subgrupo, y de ellas la
  // probabilidad de la fila i (la fila i va en la palabra i/4, componente i%4).
  const pLee = (u) => Array.from({ length: PV }, (_, w) =>
    `      let pw${u}_${w} = ps[(sgi * 32u + j0 + ${u}u) * ${PV}u + ${w}u];`).join('\n');
  const pDe = (i, u) => (psh ? `pw${u}_${i >> 2}.${'xyzw'[i & 3]}` : `subgroupShuffle(e${i}, j0 + ${u}u)`);

  // Bucle de reparto entre los hilos del grupo. Si el total se reparte exacto,
  // con un número FIJO de vueltas por hilo: el compilador puede desenrollarlo y
  // lanzar todas las lecturas de golpe.
  const reparto = (total, cuerpo) => (total % WG === 0
    ? `for (var r = 0u; r < ${total / WG}u; r = r + 1u) {
      let i = lid.x + r * ${WG}u;
${cuerpo}
    }`
    : `for (var i = lid.x; i < ${total}u; i = i + ${WG}u) {
${cuerpo}
    }`);

  return /* wgsl */`enable subgroups;
const G : u32 = ${G}u;
const TQ : u32 = ${TQ}u;
const CPR : u32 = ${CPR}u;
const QV : u32 = ${QV}u;
const NEG : f32 = -1.0e30;
// «ventana» > 0: la clave k cuenta solo si qAbs − k < ventana, la misma regla
// que el softmax de siempre (kernels.nn.wgsl.js). Va en el uniform: las capas
// globales y las de ventana comparten compilado.
// «partes»: en cuántos trozos se reparte el rango de claves de cada grupo. Con
// 1 el grupo escribe la salida final; con más, sumas parciales.
struct P {
  T : u32, nPast : u32, H : u32, KVH : u32,
  ventana : u32, qTiles : u32, partes : u32, pad0 : u32,
  scale : f32, pad1 : f32, pad2 : f32, pad3 : f32,
};
@group(0) @binding(${bQ}) var<storage, read> q : array<vec4<f32>>;     // [T][H][D]
@group(0) @binding(${bK}) var<storage, read> k : array<vec4<u32>>;     // caché [ctx][KVH][D], f16
@group(0) @binding(${bV}) var<storage, read> v : array<${VT}>;     // caché [ctx][KVH][D], f16
${conPuerta ? `@group(0) @binding(${bG}) var<storage, read> puerta : array<f32>;   // [T][H]` : ''}
${parcial
    ? `@group(0) @binding(${bPO}) var<storage, read_write> parcO : array<vec4<f32>>;   // [T][H][partes][D]
@group(0) @binding(${bPM}) var<storage, read_write> parcML : array<vec2<f32>>;  // [T][H][partes] (máximo, suma)`
    : `@group(0) @binding(${bOut}) var<storage, read_write> salida : array<vec4<f32>>;  // [T][H][D]`}
@group(0) @binding(${bP}) var<uniform> p : P;

// La tesela de K, con los trozos de cada fila permutados (ver la cabecera).
var<workgroup> ks : array<vec4<u32>, ${FLASH_BK * CPR}u>;
${qsh ? `// Las G·TQ filas de q del grupo, en f32: se leen una vez y sirven a todas las teselas.
var<workgroup> qs : array<vec4<f32>, ${R * QV}u>;` : ''}
${vsh ? `var<workgroup> vs : array<${VT}, ${FLASH_BK * 32}u>;` : ''}${psh ? `
// Las probabilidades de la tesela: [subgrupo][clave][${PV} vec4], las ${RS} filas del subgrupo.
var<workgroup> ps : array<vec4<f32>, ${NSG * FLASH_BK * PV}u>;` : ''}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(subgroup_invocation_id) lane : u32) {
  // Todo lo que decide bucles y barreras sale del uniform y del índice de grupo:
  // es uniforme en el grupo, que es lo que exige el análisis de uniformidad.
  let idx = wid.x + wid.y * 65535u;
  let parte = idx % p.partes;
  let resto = idx / p.partes;
  let kvh = resto / p.qTiles;
  if (kvh >= p.KVH) { return; }
  // Las teselas de consultas van de la ÚLTIMA a la primera: con máscara causal
  // las últimas tienen más claves, y lanzarlas antes reparte mejor la cola.
  let qt = p.qTiles - 1u - (resto % p.qTiles);
  let tIni = qt * TQ;
  let tUlt = min(tIni + TQ, p.T) - 1u;
  // Rango de claves del grupo: hasta la causal de su ÚLTIMA consulta y desde el
  // principio de la ventana de la PRIMERA. Lo de fuera no se visita.
  let kFin = p.nPast + tUlt + 1u;
  let qIni = p.nPast + tIni;
  var kIni = 0u;
  if (p.ventana > 0u && qIni + 1u > p.ventana) { kIni = qIni + 1u - p.ventana; }
  // Esta parte del rango, en múltiplos de la tesela. Puede salir vacía (rango
  // corto y muchas partes): entonces no recorre nada y deja el neutro (m, l, o)
  // = (NEG, 0, 0), que la reducción pesa con cero.
  let trozo = ((kFin - kIni + p.partes - 1u) / p.partes + ${FLASH_BK - 1}u) / ${FLASH_BK}u * ${FLASH_BK}u;
  let pIni = min(kIni + parte * trozo, kFin);
  let pFin = min(pIni + trozo, kFin);

  let sgi = lid.x / 32u;
${F.map((i) => `  let fr${i} = sgi * ${RS}u + ${i}u;                 // fila del grupo
  let ft${i} = tIni + fr${i} / G;                    // su token
  let fh${i} = kvh * G + fr${i} % G;                 // su cabeza de consulta
  let fa${i} = p.nPast + ft${i};                     // su posición absoluta
  let qb${i} = ${qsh ? `fr${i} * QV` : `(min(ft${i}, p.T - 1u) * p.H + fh${i}) * QV`};
  var m${i} = NEG;
  var l${i} = 0.0;
${E.map((e) => `  var o${i}_${e} = vec4<f32>(0.0);`).join('\n')}`).join('\n')}
${qsh ? `
  // q a memoria de grupo. Las filas de tokens que no existen (última tesela
  // incompleta) se rellenan con ceros: se calculan y no se escriben.
  for (var i = lid.x; i < ${R * QV}u; i = i + ${WG}u) {
    let r = i / QV;
    let t = tIni + r / G;
    var val = vec4<f32>(0.0);
    if (t < p.T) { val = q[(t * p.H + kvh * G + r % G) * QV + i % QV]; }
    qs[i] = val;
  }` : ''}

${pref ? `
  // Prelectura: la primera tesela de K, a registros (ver «pref»).
${Array.from({ length: NR }, (_, r) => `  var kp${r} = vec4<u32>(0u);`).join('\n')}
  if (pIni < pFin) {
${Array.from({ length: NR }, (_, r) => `    kp${r} = k[(min(pIni + (lid.x + ${r * WG}u) / CPR, pFin - 1u) * p.KVH + kvh) * CPR + (lid.x + ${r * WG}u) % CPR];`).join('\n')}
  }` : ''}

  var kb = pIni;
  loop {
    if (kb >= pFin) { break; }
    // WAR: nadie puede pisar la tesela anterior mientras alguien la lee. En la
    // primera vuelta hace además de RAW para las filas de q.
    workgroupBarrier();
    // Las filas de la tesela que pasan del final se recortan a la última clave
    // válida: se leen datos de verdad (finitos) y la máscara las deja a cero.
    ${pref
    ? `${Array.from({ length: NR }, (_, r) => `{
      let i = lid.x + ${r * WG}u;
      ks[(i / CPR) * CPR + ((i % CPR) ^ ((i / CPR) & ${SWZ}u))] = kp${r};
    }`).join('\n    ')}`
    : reparto(FLASH_BK * CPR, `      let j = i / CPR;
      let c = i % CPR;
      let fila = min(kb + j, pFin - 1u) * p.KVH + kvh;
      ks[j * CPR + (c ^ (j & ${SWZ}u))] = k[fila * CPR + c];`)}${vsh ? `
    ${reparto(FLASH_BK * 32, `      let fila = min(kb + i / 32u, pFin - 1u) * p.KVH + kvh;
      vs[i] = v[fila * 32u + i % 32u];`)}` : ''}
    workgroupBarrier();${pref ? `
    // La tesela siguiente, pedida ya: llega mientras se calcula esta.
    if (kb + ${FLASH_BK}u < pFin) {
${Array.from({ length: NR }, (_, r) => `      kp${r} = k[(min(kb + ${FLASH_BK}u + (lid.x + ${r * WG}u) / CPR, pFin - 1u) * p.KVH + kvh) * CPR + (lid.x + ${r * WG}u) % CPR];`).join('\n')}
    }` : ''}

    // ── Q·Kᵀ: este carril es la clave kb + lane ─────────────────────────────
    let clave = kb + lane;
    let kFila = lane * CPR;
    let kPerm = lane & ${SWZ}u;
${F.map((i) => `    var s${i} = 0.0;`).join('\n')}
    for (var c = 0u; c < CPR; c = c + 1u) {
      let kw = ks[kFila + (c ^ kPerm)];
      let k0 = vec4<f32>(unpack2x16float(kw.x), unpack2x16float(kw.y));
      let k1 = vec4<f32>(unpack2x16float(kw.z), unpack2x16float(kw.w));
${F.map((i) => `      s${i} = s${i} + dot(${qLee(i, '2u * c')}, k0) + dot(${qLee(i, '2u * c + 1u')}, k1);`).join('\n')}
    }

    // ── softmax en línea, por fila ──────────────────────────────────────────
    // Misma máscara y misma escala que el softmax de siempre: la clave cuenta si
    // está en esta parte, no pasa de la consulta y cae dentro de la ventana. Lo
    // que no cuenta entra al máximo como NEG y a la suma como cero exacto.
    let enParte = clave < pFin;
${F.map((i) => `    let vis${i} = enParte && clave <= fa${i} && (p.ventana == 0u || fa${i} - clave < p.ventana);
    let x${i} = select(NEG, s${i} * p.scale, vis${i});
    let mn${i} = max(m${i}, subgroupMax(x${i}));
    // exp(m − mn) ≤ 1: si el máximo sube, lo acumulado se reescala; si no, ×1.
    let al${i} = exp(m${i} - mn${i});
    let e${i} = select(0.0, exp(x${i} - mn${i}), vis${i});
    l${i} = l${i} * al${i} + subgroupAdd(e${i});
    m${i} = mn${i};
${E.map((e) => `    o${i}_${e} = o${i}_${e} * al${i};`).join('\n')}`).join('\n')}${psh ? `
    // Las ${RS} probabilidades de la clave de este carril, a memoria de grupo; la
    // barrera las hace visibles. La de arriba del bucle protege ps de la tesela
    // siguiente mientras alguien aún lee esta.
${Array.from({ length: PV }, (_, w) => `    ps[(sgi * 32u + lane) * ${PV}u + ${w}u] = vec4<f32>(${[0, 1, 2, 3].map((c) => (4 * w + c < RS ? `e${4 * w + c}` : '0.0')).join(', ')});`).join('\n')}
    workgroupBarrier();` : ''}

    // ── P·V: este carril son las dimensiones [${DPL}·lane, ${DPL}·lane + ${DPL}) ─────────
    // La probabilidad de la clave kb + j vive en el carril j: la reparte el
    // subgroupShuffle. Las claves sin probabilidad (fuera de la máscara o
    // recortadas) suman un cero exacto.
    for (var j0 = 0u; j0 < ${FLASH_BK}u; j0 = j0 + ${U}u) {
${Us.map(vLee).join('\n')}
${Us.map(vvDe).join('\n')}${psh ? '\n' + Us.map(pLee).join('\n') : ''}
${F.map((i) => Us.map((u) => `      {
        let pj = ${pDe(i, u)};
${E.map((e) => `        o${i}_${e} = o${i}_${e} + pj * vv${u}_${e};`).join('\n')}
      }`).join('\n')).join('\n')}
    }
    kb = kb + ${FLASH_BK}u;
  }

${F.map((i) => (parcial
    ? `  if (ft${i} < p.T) {
    let pb${i} = (ft${i} * p.H + fh${i}) * p.partes + parte;
${E.map((e) => `    parcO[pb${i} * QV + lane * ${NV4}u + ${e}u] = o${i}_${e};`).join('\n')}
    if (lane == 0u) { parcML[pb${i}] = vec2<f32>(m${i}, l${i}); }
  }`
    : `  if (ft${i} < p.T) {
    // Siempre hay al menos una clave visible (la propia posición), así que l > 0.
    let inv${i} = 1.0 / l${i};
    let ob${i} = (ft${i} * p.H + fh${i}) * QV + lane * ${NV4}u;
${conPuerta ? `    // La puerta por cabeza de lfm3, con la MISMA expresión que \`mulCabezaShader\`.
    let den${i} = 1.0 + exp(-puerta[ft${i} * p.H + fh${i}]);
` : ''}${E.map((e) => `    salida[ob${i} + ${e}u] = ${conPuerta ? `(o${i}_${e} * inv${i}) / den${i}` : `o${i}_${e} * inv${i}`};`).join('\n')}
  }`)).join('\n')}
}`;
}

// Junta las partes del rango de claves de cada fila (t, h): el máximo común,
// los pesos exp(m_parte − M), la suma y la salida normalizada, con la puerta si
// la hay. Un hilo por vec4 de salida. Las partes vacías traen (NEG, 0, 0) y
// pesan exactamente cero.
export function flashReduceShader({ D, puerta = false }) {
  if (![128, 256].includes(D)) throw new Error(`flash: headDim ${D} sin soporte (128 o 256)`);
  const QV = D / 4;
  const bG = puerta ? 2 : -1;
  const bOut = puerta ? 3 : 2;
  const bP = bOut + 1;
  return /* wgsl */`
const QV : u32 = ${QV}u;
const NEG : f32 = -1.0e30;
struct P { filas : u32, partes : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read> parcO : array<vec4<f32>>;    // [T·H][partes][D]
@group(0) @binding(1) var<storage, read> parcML : array<vec2<f32>>;   // [T·H][partes]
${puerta ? `@group(0) @binding(${bG}) var<storage, read> puerta : array<f32>;   // [T][H] = [T·H]` : ''}
@group(0) @binding(${bOut}) var<storage, read_write> salida : array<vec4<f32>>;   // [T][H][D]
@group(0) @binding(${bP}) var<uniform> p : P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 64u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.filas * QV) { return; }
  let fila = i / QV;
  let base = fila * p.partes;
  var mx = NEG;
  for (var j = 0u; j < p.partes; j = j + 1u) { mx = max(mx, parcML[base + j].x); }
  var suma = 0.0;
  var o = vec4<f32>(0.0);
  for (var j = 0u; j < p.partes; j = j + 1u) {
    let ml = parcML[base + j];
    let w = exp(ml.x - mx);
    suma = suma + ml.y * w;
    o = o + parcO[(base + j) * QV + i % QV] * w;
  }
  // suma > 0 siempre que la fila tenga alguna clave visible, que la tiene; el
  // select solo evita un NaN si algún día no fuera así.
  let r = o * select(0.0, 1.0 / suma, suma > 0.0);
  salida[i] = ${puerta ? 'r / (1.0 + exp(-puerta[fila]))' : 'r'};
}`;
}
