// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Rotación de Hadamard sobre las activaciones — para pesos en base rotada.
// ─────────────────────────────────────────────────────────────────────────────
// Algunos modelos ternarios guardan los pesos girados por una rotación ortogonal
// y esperan que el runtime aplique la misma rotación a las activaciones. No es
// opcional: sin ella el modelo no da texto malo, da ruido. El fichero declara la
// rotación en sus metadatos, así que un runtime o la aplica o se niega a cargar.
//
// LA MATRIZ, tal cual la construye la implementación de referencia:
//     H[r][c] = (-1)^popcount(r & c) / sqrt(N)
// Pese a que el fichero la llama «walsh», ese es el orden NATURAL de Sylvester,
// y eso importa mucho: la transformada RÁPIDA en orden natural la calcula
// exacta, así que el coste es log2(N) etapas de mariposa en vez de una
// multiplicación por una matriz de N×N. Con N=1024 son 10 etapas contra un
// millón de multiplicaciones por bloque. Comprobado contra la matriz explícita.
//
// Es SIMÉTRICA y ORTOGONAL, luego es su propia inversa: H(Hx) = x. Por eso la
// tabla de embeddings —que guarda las filas ya giradas— se deshace con la misma
// transformada, solo que con los signos en el otro orden.
//
// DOS ÓRDENES, y confundirlos da ruido silencioso:
//   · directo (antes de un matmul):   y = H(s ⊙ x)   signos PRIMERO
//   · inverso (tras el embedding):    y = s ⊙ (H x)  signos DESPUÉS
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE).

// `bloque` es el tamaño de la rotación (1024 en los modelos que la usan) y
// `wg` cuántos hilos la procesan. Cada workgroup se queda un bloque entero en
// memoria compartida: 1024 f32, holgado frente al mínimo que garantiza WebGPU.
export function hadamardShader(bloque = 1024, wg = 256) {
  const etapas = Math.log2(bloque);
  if (!Number.isInteger(etapas)) throw new Error(`bloque de Hadamard no es potencia de dos: ${bloque}`);
  return /* wgsl */`
const WG : u32 = ${wg}u;
const N  : u32 = ${bloque}u;
const MITAD : u32 = ${bloque / 2}u;
// La normalización 1/sqrt(N) se aplica UNA vez al final, no por etapa: así la
// transformada rápida hace solo sumas y restas y no arrastra redondeos.
const ESCALA : f32 = ${(1 / Math.sqrt(bloque)).toString()};

struct P {
  ancho    : u32,   // elementos por fila (múltiplo de N)
  filas    : u32,
  porFila  : u32,   // ancho / N
  inverso  : u32,   // 0 = signos antes (directo) · 1 = signos después
  permHd   : u32,   // reordenado de rasgos: 0 = ninguno
  permNk   : u32,
  permRep  : u32,
  pad      : u32,
};
@group(0) @binding(0) var<storage, read>       src   : array<f32>;
@group(0) @binding(1) var<storage, read>       signo : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst   : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;

var<workgroup> s : array<f32, ${bloque}>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let grupo = wid.x + wid.y * 65535u;        // dos ejes: uno solo topa en 65535
  if (grupo >= p.filas * p.porFila) { return; }
  let fila = grupo / p.porFila;
  let trozo = grupo % p.porFila;
  let base = fila * p.ancho + trozo * N;     // dónde empieza este bloque
  let baseSg = trozo * N;                    // el signo va por POSICIÓN en la fila

  // Carga. En el camino directo los signos se aplican AQUÍ, antes de rotar.
  for (var i = lid.x; i < N; i = i + WG) {
    // REORDENADO DE RASGOS, cuando el fichero lo pide para este tensor. La
    // activación viene por teselas [hd, nk, rep] y el peso se guardó agrupado
    // [hd, rep, nk]. Mezcla ENTRE bloques de N, así que no puede hacerse
    // después de trocear: se hace aquí, al leer, y sale gratis.
    //
    // Se calcula el índice ORIGEN a partir del destino, que es lo que necesita
    // una carga: destino p = hd + Hd*(rep + Rep*nk) -> origen hd + Hd*(nk + Nk*rep).
    var orig = trozo * N + i;
    if (p.permRep > 1u) {
      let pos = trozo * N + i;
      let hd = pos % p.permHd;
      let q  = pos / p.permHd;
      // El sentido del reordenado es un 50/50 que NO se deduce de la lectura:
      // hay que probarlo. El interruptor __ELFFUSS_PERM_INV invierte el par.
      let rep = select(q % p.permRep, q / p.permNk, p.pad == 1u);
      let nk  = select(q / p.permRep, q % p.permNk, p.pad == 1u);
      orig = hd + p.permHd * (nk + p.permNk * rep);
    }
    let v = src[fila * p.ancho + orig];
    s[i] = select(v * signo[baseSg + i], v, p.inverso == 1u);
  }
  workgroupBarrier();

  // Transformada rápida en orden natural. Las N/2 mariposas de cada etapa tocan
  // índices DISJUNTOS, así que dentro de una etapa no hace falta barrera; entre
  // etapas sí. El bucle de etapas es uniforme, que es lo que exige la barrera.
  for (var et = 0u; et < ${etapas}u; et = et + 1u) {
    let len = 1u << et;
    for (var b = lid.x; b < MITAD; b = b + WG) {
      let j = (b & (len - 1u)) | ((b >> et) << (et + 1u));
      let a = s[j];
      let c = s[j + len];
      s[j] = a + c;
      s[j + len] = a - c;
    }
    workgroupBarrier();
  }

  // Escritura. En el camino inverso los signos se aplican AQUÍ, después.
  for (var i = lid.x; i < N; i = i + WG) {
    let v = s[i] * ESCALA;
    dst[base + i] = select(v, v * signo[baseSg + i], p.inverso == 1u);
  }
}`;
}
