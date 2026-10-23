# Elffuss Engine — runtime de inferencia propio

Motor de inferencia LLM en el navegador (WebGPU) + empaquetador offline de pesos.
Sustituye a `@litert-lm/core` en Elffuss Claw/Code.

Elffuss Engine vive ahora en la organización UtopiaIA de GitHub, en github.com/utopia-ia/elffuss-engine. Se movió desde github.com/KikoCis/elffuss-engine el 23 de octubre de 2026; la dirección antigua redirige.

> **El motor (`js/`) es Apache-2.0** y se publica en
> `github.com/utopia-ia/elffuss-engine`. El empaquetador de pesos y el compresor
> NO viven en este repositorio y no se publican. Ver `LICENSE`. El plan y el
> porqué están en `ENGINE-PLAN.md`.
>
> Este repositorio (Gitea) es donde se trabaja; el de GitHub es el espejo
> publicado. Antes de empujar allí, comprobar que no sale nada que no deba:
> rutas de disco, medidas absolutas de la máquina ni recetas de cuantización.

## Qué vive aquí y qué no

| | dónde | licencia |
|---|---|---|
| Motor WebGPU (`js/`) | este repo + GitHub público | Apache-2.0 |
| Bancos (`tests/`) | este repo | Apache-2.0 |
| Empaquetador / compresor | **fuera de este repo** | no se publica |
| App Elffuss Claw (chat, agente, UI) | `../elffuss-claw`, GitHub público | Apache-2.0 |

## Qué protege y qué no un repo de código de navegador

El `.js` que se sirve al navegador lo lee cualquiera con las devtools abiertas:
eso es física del medio, no una decisión de este repo. Pretender que el motor
era cerrado mientras se servía en tres sitios públicos era una postura que el
medio no sostiene, y por eso `js/` es Apache-2.0.
Lo que sí queda fuera del alcance de cualquiera es el empaquetador (corre en
nuestra máquina, no viaja al cliente), y ahí es donde conviene que esté el
trabajo que no se quiere regalar: la receta de cuantización y la calibración.

## Cómo se enchufa en la app pública

El motor **no se commitea** en `elffuss-claw`. Se copia al árbol de trabajo
local y de ahí sale por rsync en el deploy, igual que ya se hace con los pesos
de `web/models/`:

```bash
./sync-to-claw.sh                  # js/ → ../elffuss-claw/web/js/engine/  (gitignorado allí)
./sync-to-claw.sh ../elffuss-code  # y a code.elffuss, que ofrece los mismos modelos
```

## Estructura

```
js/       módulos del runtime (ES modules vanilla, sin deps ni bundler)
          · registro.js — qué modelos hay, dónde viven y `disponible(id)`.
            Va SEPARADO de provider.js porque la app necesita sondear
            disponibilidad sin bajarse el motor entero, y porque tener las
            URLs en dos sitios ya dejó al selector mirando un fichero muerto.
rust/     empaquetador/compresor offline (CLI, nunca viaja al navegador)
tests/    tests de Node puro (parser, layouts) — sin GPU ni DOM
ref/      volcados de verdad de referencia contra los que asertan los tests
```

## Verificación

```bash
npm run check                # sintaxis de todo
npm test                     # parser GGUF: fixture + modelos reales de disco
npm run test:quant-cpu       # los 8 tipos, referencia de CPU vs llama.cpp (sin GPU)
npm run test:quant           # los 8 tipos en la GPU: dequant, matmul y gather
npm run test:tiny            # forward completo con seis tipos MEZCLADOS
npm run test:gpu             # ... y el resto, con --model si se tiene el fichero
```

### Cómo se validan los tipos cuantizados

Ocho: `F32`, `F16`, `Q8_0` (el 0.8B) y los cinco del 27B — `IQ1_S`, `IQ2_XXS`,
`Q2_K`, `Q4_K`, `Q5_K`. Los ocho comparten andamio de shader y se diferencian
solo en el desempaquetado, así que lo que hay que probar es exactamente eso.

**La descuantización se exige BIT A BIT contra llama.cpp**, no con tolerancia.
La verdad la genera `gguf-py` — el descodificador del propio llama.cpp — sobre
bloques sintéticos deterministas (`ref/dump-quant-truth.py`). Es deliberado que
la verdad venga de fuera: la referencia de CPU en JS y el shader WGSL los
escribe la misma cabeza, y un layout mal entendido se cancelaría consigo mismo.

El único margen que se concede: en `Q2_K`/`Q4_K`/`Q5_K` el peso sale de `d·q − m`
y IEEE permite a la GPU fundir la multiplicación y la resta en un FMA. Las dos
formas son legales y dan bits distintos, así que la referencia calcula **las
dos** y se acepta cualquiera. Sigue siendo igualdad exacta, contra el conjunto
de resultados que el estándar permite.

El **matmul** se compara contra una referencia en f64 (los `Number` de
JavaScript son dobles) con NMSE y con el error máximo medido **contra la escala
de la salida**, no contra cada elemento: una fila cuyos sumandos se cancelan da
un resultado diminuto, y ahí un error absoluto normal se convierte en un
relativo enorme que no dice nada del kernel.

La referencia se calcula siempre con **los mismos pesos que usa la GPU**, ya
descuantizados. Compararla contra los f32 de antes de cuantizar mediría el error
de cuantización (~1e-3), que sepulta el del kernel y deja el test inútil.

**`test:tiny` es el que responde la pregunta del 27B.** Fabrica en memoria un
qwen35 de juguete (`tests/synth-gguf.mjs`) con la misma forma que el real y con
los seis tipos repartidos entre sus tensores, y lo escribe DOS veces: cuantizado
y en F32 con los pesos ya descuantizados. Los dos tienen que dar los mismos
logits. Eso no comprueba que el modelo cargue, comprueba que el camino
cuantizado *calcula lo mismo* hasta la salida — un tensor leído con el tipo
equivocado sobrevive a «carga sin error», a esto no.

Los tests de GPU corren **dentro de Chromium**, no en Node: los límites, las
features y los bugs que importan son los del navegador. `tests/browser.mjs`
sirve el repo por `http://127.0.0.1` (con soporte de `Range`, que es como el
motor lee los pesos), abre Chromium y ejecuta el módulo de test en la página.
Reutiliza el playwright de `../elffuss-claw/tests` — no se instala nada aquí; si
no lo encuentra, se salta en voz alta. Se puede forzar con `PLAYWRIGHT_CORE=…`,
`CHROMIUM_PATH=…`, `HEADED=1` y `VERBOSE=1`.

**Dos trampas que el arnés cierra** (las dos hacen que parezca un fallo de GPU
cuando no lo es):

- `about:blank` **no es contexto seguro** → `navigator.gpu` sale `undefined`.
  Por eso hay servidor y no un `setContent()`.
- Con `--enable-unsafe-webgpu` y **sin** fijar el backend gráfico nativo,
  Chromium devuelve un adaptador **SwiftShader**: CPU pura disfrazada de GPU. Funciona, pasa los
  tests y falsea cualquier medición. El arnés aborta si detecta software.

Los tests comparan contra una **verdad de referencia** sacada del lector de
llama.cpp, no de nuestro parser — un test que compara el código consigo mismo no
prueba nada. Para regenerarla (o añadir modelos):

```bash
PYTHONPATH=$HOME/llama.cpp/gguf-py OUT=ref/gguf-truth.json \
  python3 ref/dump-truth.py <modelo.gguf> [más modelos…]
```

El volcado (`ref/*.json`) está gitignorado: pesa y se regenera. Si falta, los
casos exhaustivos se saltan y lo dicen en voz alta.
