// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · matmul del decode con VARIAS FILAS por workgroup.
// ─────────────────────────────────────────────────────────────────────────────
// POR QUÉ EXISTE ESTE FICHERO. El kernel de `kernels.wgsl.js` hace UNA fila por
// workgroup, y esa decisión tiene un precio que no se ve hasta que se cuenta el
// tráfico:
//
//   · por cada peso se leen 50/256 de byte de peso (IQ1_S)…
//   · …y 4 bytes de `x`, la activación en f32 que le corresponde.
//
// No es una errata. Cada workgroup se lee la fila ENTERA de activaciones, y hay
// una fila de activaciones leída por cada fila de salida: `x` se vuelve a leer
// `rows` veces por matmul. Son 20,5 bytes de activación por cada byte de peso.
// Eso cuadra con lo que midió el oráculo __ELFFUSS_X_FIJA (traer `x` = 35,6% del
// matmul, contra 6,2% de traer pesos+rejilla).
//
// La dirección de `x` NO depende de la fila: en el reparto por carril vale
// `(base_t*k + k0 + obase) >> 2`, donde `obase` sale de `lid.x` y `k0` de `b`.
// O sea que un carril puede leer sus dos vec4 UNA vez y usarlas contra los pesos
// de N filas distintas. Con N filas por workgroup:
//
//   · las lecturas de `x` se dividen entre N (con N=4, la cuarta parte),
//   · los lanzamientos de workgroup se dividen entre N,
//   · el prólogo (leer params, calcular row, guardas) se paga N veces menos,
//   · la escritura de `y` deja de ser un f32 suelto por workgroup y pasan a ser
//     N f32 CONTIGUOS desde el mismo carril,
//   · y dentro del bucle hay N cadenas de dependencia INDEPENDIENTES (gi -> tbl
//     -> fma) en vuelo a la vez, que es paralelismo de memoria sin pedir más
//     ocupancia.
// Lo que NO cambia: los bytes de peso leídos, la aritmética por peso y el orden
// de acumulación.
//
// ES BIT A BIT EXACTO, por construcción: cada fila acumula en el mismo carril,
// en el mismo orden de `b` y de `j`, y se reduce con el mismo `subgroupAdd` de
// 32 carriles. Lo único que cambia es CUÁNTAS filas lleva un workgroup. Se puede
// validar con igualdad EXACTA contra el kernel de una fila, no con tolerancia.
//
// AVISO DE MANTENIMIENTO: el cuerpo por fila de aquí es una COPIA del de
// `IQ1_S.decodeCarril` en `kernels.wgsl.js` (mismas nueve líneas, mismo orden).
// Se copia y no se reutiliza porque `decodeCarril` emite también el prólogo
// (`ib`, `l`, `obase`) y la lectura de `x`, que aquí van IZADOS fuera del bucle
// de filas — que es justamente el punto. Si se toca el descuantizador allí, hay
// que tocarlo aquí: la prueba de cuantización lo caza, porque valida el matmul
// contra la referencia de CPU.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import { HELPERS, AXIS, quantInfo } from './kernels.wgsl.js';

// Cuántas filas lleva un workgroup. Devuelve 1 (= kernel de siempre) salvo que
// se cumpla TODO: interruptor puesto, tipo con reparto por carril, workgroup de
// 32, decode (tile 1), y `rows` múltiplo del reparto.
//
// El múltiplo NO es comodidad: sin él haría falta una cola con guarda por fila
// DENTRO del bucle de bloques, que es justo la rama que se quiere quitar. Las
// formas del decode del 27B son todas múltiplo de 8 (17408, 10240, 6144, 5120,
// 2048, 248320), así que la condición no descarta nada que importe.
//
// El suelo de 2048 filas es por PARALELISMO: con `rows` pequeño (ssm_alpha y
// ssm_beta son 48 filas) dividir entre N deja al despacho sin workgroups para
// llenar el dispositivo, y esas dos formas no pesan nada en el token.
//
// VIVE AQUÍ Y LO IMPORTA ops.js, como AXIS: si el despacho y el shader
// calcularan el reparto por su cuenta y se desincronizaran, saldrían filas a
// cero sin un solo error.
export function filasPorGrupo(type, workgroupSize, tile, carrilSubbloque, rows, subgroup) {
  const q = Math.max(1, globalThis.__ELFFUSS_FILAS_WG | 0);
  if (q === 1) return 1;
  // El shader emite `enable subgroups;` y `subgroupAdd`. Sin subgrupos el
  // pipeline NO COMPILA, y `threadsFor` devuelve 32 tambien en esa ruta para
  // bpr=20 y bpr=24 — las dos formas que mas pesan. Sin esta guarda, poner
  // `__ELFFUSS_NO_SUBGROUP` (lo hacen dos de las pruebas internas)
  // reventaria la creacion del pipeline en vez de caer al kernel de una fila.
  if (!subgroup) return 1;
  const d = quantInfo(type);
  const porCarril = carrilSubbloque && workgroupSize === 32
    && typeof d.decodeCarril === 'function';
  if (!porCarril || tile !== 1) return 1;
  // Suelo de paralelismo, configurable. Con 2048 fijo, el modelo de juguete
  // (filas de decenas) nunca toma esta ruta y haria falta cargar el 27B solo
  // para saber si compila. En produccion el defecto sigue siendo 2048, asi que
  // ssm_alpha y ssm_beta (48 filas) se quedan fuera igual.
  const suelo = (globalThis.__ELFFUSS_FILAS_MIN | 0) || 2048;
  if (rows < suelo || rows % q !== 0) return 1;
  return q;
}

// y[fila] = Σ_k A[fila][k] · x[k] para `filas` filas consecutivas por workgroup.
// Solo decode (tokens = 1, tile = 1) y solo tipos con reparto por carril.
export function matmulFilasShader(type, filas, sgSize = 32) {
  const d = quantInfo(type);
  if (typeof d.decodeCarril !== 'function' || !d.table) {
    // Sin esto, un tipo nuevo se llevaría un shader que no le corresponde y el
    // bind group seguiría siendo válido: filas de ceros sin un solo error.
    throw new Error(`matmulFilasShader: «${type}» no tiene reparto por carril`);
  }
  if (!(filas > 1)) throw new Error('matmulFilasShader: filas debe ser > 1');
  const COMP = ['x', 'y', 'z', 'w'];

  // Cuerpo de UNA fila. Copia literal de IQ1_S.decodeCarril sin su prólogo:
  // `ib`, `l`, `obase`, `xb`, `xv0` y `xv1` están izados fuera.
  const fila = (r) => `      {
        let blk = ((row0 + ${r}u) * bpr${globalThis.__ELFFUSS_W_FIJA === true ? '' : ' + b'}) * ${d.blockBytes}u;
        let dq = load_scale(blk);
        let qh = load_u16_at(blk + 34u + ib * 2u);
        let dl = dq * f32(2u * ((qh >> 12u) & 7u) + 1u);
        let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
        let gi = load_byte(blk + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);
        let g0 = tbl[gi * 2u];
        let g1 = tbl[gi * 2u + 1u];
${Array.from({ length: 8 }, (_, j) => `        {
          let gv = f32(get_byte_i32(${j < 4 ? 'g0' : 'g1'}, ${j % 4}u));
          let wv = dl * (gv + delta);
          acc${r} = acc${r} + wv * ${j < 4 ? 'xv0' : 'xv1'}.${COMP[j % 4]};
        }`).join('\n')}
      }`;

  return /* wgsl */`enable subgroups;

const FILAS : u32 = ${filas}u;
struct Params { nBlocks: u32, blocksPerRow: u32, rows: u32, k: u32,
                tokens: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> w : array<u32>;
@group(0) @binding(1) var<storage, read> x : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;
@group(0) @binding(4) var<storage, read> tbl : array<u32>;
${HELPERS}
@compute @workgroup_size(${sgSize})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  // Dos ejes de filas, igual que el kernel de una fila, pero ahora cada punto
  // de la rejilla son FILAS filas consecutivas.
  let row0 = (wid.x + wid.y * ${AXIS}u) * FILAS;
  let base_t = wid.z;
  let bpr = params.blocksPerRow;
  // Prólogo IZADO: no depende ni de la fila ni del bloque.
  let ib = lid.x >> 2u;
  let l = lid.x & 3u;
  let obase = ib * 32u + l * 8u;

${Array.from({ length: filas }, (_, r) => `  var acc${r} = 0.0;`).join('\n')}

  if (row0 < params.rows) {
    for (var b = 0u; b < bpr; b = b + 1u) {
      let k0 = b * ${d.blockElems}u;
      // Las DOS vec4 de activaciones del carril, leídas UNA vez para las FILAS
      // filas. Aquí es donde se paga o no se paga el 35,6%.
      let xb = (base_t * params.k${globalThis.__ELFFUSS_X_FIJA === true ? '' : ' + k0 + obase'}) >> 2u;
      let xv0 = x[xb];
      let xv1 = x[xb + 1u];
${Array.from({ length: filas }, (_, r) => fila(r)).join('\n')}
    }
  }

  // Una reducción por fila. Son FILAS por workgroup en vez de una, pero hay
  // FILAS veces menos workgroups: el total de subgroupAdd no cambia. Lo que sí
  // cambia es que los FILAS f32 que escribe el carril 0 son CONTIGUOS.
${Array.from({ length: filas }, (_, r) => `  {
    let tot${r} = subgroupAdd(acc${r});
    if (lid.x == 0u && row0 + ${r}u < params.rows) {
      y[base_t * params.rows + row0 + ${r}u] = tot${r};
    }
  }`).join('\n')}
}`;
}
