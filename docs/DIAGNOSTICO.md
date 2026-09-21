# Diagnóstico inicial (Fase 1)

Fecha: 2026-09-20

> **Actualización (mismo día).** El dataset se capturó y se importó al
> proyecto: 100 cilindros y 599 fotografías. Lo que sigue describe el estado
> en que se encontró el proyecto antes de esa importación, y se conserva
> porque las conclusiones técnicas que motivaron el diseño siguen vigentes.
> El contenido real del dataset está en [DATASET.md](DATASET.md).
>
> Predicciones de este diagnóstico que el dataset confirmó:
> el troquelado en metal (aunque en el **hombro**, no en el collarín),
> la diversidad de formatos de serial (12 patrones distintos, lo que valida
> no fijar una máscara escrita a mano) y las seis condiciones de captura.
> Lo que no anticipó: que `PH` contiene una **periodicidad** ("CADA 10 AÑOS")
> en 64 de los 100 cilindros, y que hay un serial duplicado.

## 1. Qué se encontró

### Proyecto

El directorio de trabajo `~/Desktop/npl_ai` estaba **vacío**: sin código, sin
configuración, sin control de versiones.

### Dataset

**No existe todavía ninguna fotografía de cilindros**, ni en el equipo local
ni en Drive. Lo que hay en la carpeta compartida "Cilindros Electrametal"
(propiedad de ruizpoloerickayadira@gmail.com) es un esqueleto de carpetas:

| Carpeta                      | Contenido real            |
| ---------------------------- | ------------------------- |
| `Cilindro_002/5.Luz normal/` | vacía                     |
| `Cilindro_002/6.Poca luz/`   | vacía                     |
| `Cilindro_002/Descripción/`  | `Descripción.docx` (9 KB) |
| `Cilindro_003/`              | vacía                     |
| `Cilindro_025/`              | vacía                     |

Se verificó también el disco local (Escritorio, Descargas, Documentos,
Imágenes, iCloud) y el resto del Drive del usuario: ninguna fotografía de
cilindros.

### Documentación del proyecto

Sí se localizó el informe de investigación, `LIMA-NORTE_PI_IPARRAGUIRRE_RUIZ.pdf`
(Iparraguirre Vilchez y Ruiz Polo, UCV, 2026), del que se extrajeron los datos
de contexto de este diagnóstico.

## 2. La única ficha de cilindro disponible

`Cilindro_002/Descripción/Descripción.docx`:

```
N° DE SERIE: 19S206055          MARCA: JP5
AÑO DE FABRICACIÓN: 09/2019     TIPO DE GAS: DIÓXIDO DE CARBONO CO2
UNIDAD DE MEDIDA: KG            PESO: 45.2 KG
ANCHO DIÁMETRO: 22 CM           LARGO ALTURA: 1.21 CM
LITROS: 40.4 L                  COLOR: GRIS
PH: 02/2025                     PROCEDENCIA: CHN
```

## 3. Hallazgos que cambian el diseño

### 3.1 El número de serie está troquelado en metal, no impreso

El informe lo dice explícitamente: el sistema debe "reconocer de forma
automática los caracteres **grabados en el metal** de los cilindros", y habla
de capturar "las **series metálicas**" de los 100 cilindros.

Es el hallazgo de mayor impacto. Leer texto estampado no se parece a leer una
etiqueta:

- carácter y fondo son el mismo metal: no hay contraste de color, sólo la
  sombra del relieve;
- la legibilidad depende del ángulo de la luz más que de su cantidad;
- el collarín es curvo, así que el texto describe un arco;
- hay óxido, pintura y suciedad encima.

PaddleOCR está entrenado sobre texto impreso y rinde peor aquí. De ahí el
módulo `app/preprocessing/enhance.py`, que convierte relieve en contraste, y
la estrategia de probar varias transformaciones y combinarlas por consenso.

### 3.2 "PH" es la Prueba Hidrostática, no acidez

En la ficha, `PH: 02/2025` es una **fecha**, no un número. Corresponde a la
última prueba hidrostática superada, que es un dato de seguridad obligatorio
en recipientes a presión.

Modelarlo como valor numérico de acidez habría producido un schema incorrecto
y un campo imposible de rellenar. Se modela como cadena `AAAA-MM`.

### 3.3 El inventario es cerrado: 100 cilindros

El informe fija una población censal de 100 cilindros. Eso cambia la
naturaleza del problema: no hay que reconocer una cadena arbitraria, sino
decidir cuál de 100 seriales conocidos aparece en la foto.

Es la palanca principal del diseño. Resolver la lectura contra un catálogo por
distancia de edición ponderada hace que un OCR imperfecto siga identificando
bien. Verificado sobre el cilindro 002: `195206055`, `19s2O6O55` y `I9S2060S5`
resuelven los tres a `19S206055`.

### 3.4 Campos que faltaban y campos que sobran

La especificación inicial pedía once campos. Frente a la ficha real:

- **Faltan** `unidad_medida` (KG) y `procedencia` (CHN). Se han añadido.
- **`m3` no aparece** en la ficha. Se mantiene en el contrato, pero debe
  venir del backend: no está troquelado.

### 3.5 Una incoherencia de unidades en el propio dato maestro

`LARGO ALTURA: 1.21 CM` es físicamente imposible para un envase de 40 litros:
son 1,21 m. El normalizador detecta valores fuera de rango, prueba las
conversiones habituales y deja constancia en `warnings`.

### 3.6 Una sola muestra no define el formato del serial

`19S206055` sugiere "2 dígitos + letra + 6 dígitos", y los dos primeros
dígitos coinciden con el año de fabricación (2019). Es una hipótesis
plausible, no una regla: con n=1 no se puede afirmar.

Por eso el validador **no** lleva ese patrón escrito a mano. Lo deriva del
catálogo, y sólo fija una máscara posicional cuando hay al menos 8 seriales
que la respaldan. Importa porque la máscara se usa para _corregir_ caracteres:
una máscara equivocada corrompería lecturas que eran correctas.

### 3.7 Discrepancia sobre el volumen del dataset

La especificación mencionaba "unas 100 fotografías por cilindro", lo que con
100 cilindros daría 10.000 imágenes. El informe sólo fija 100 cilindros, sin
decir cuántas fotos por unidad.

Capturar 10.000 fotos a mano no es realista. Para detectar una región bastan
bastantes menos, y lo que el modelo necesita es **variedad de cilindros**, no
repetición del mismo. La recomendación es **15 fotografías por cilindro**
cubriendo las seis condiciones: 1.500 imágenes, suficientes y alcanzables.

Si hubiera que elegir: 15 fotos de 100 cilindros valen mucho más que 100 fotos
de 15 cilindros.

### 3.8 El protocolo de captura ya estaba esbozado

Las carpetas se llaman `5.Luz normal` y `6.Poca luz`. La numeración implica un
plan de seis condiciones del que faltan las cuatro primeras, previsiblemente
frontal, diagonal, cerca y lejos. Se ha formalizado en
[PROTOCOLO_CAPTURA.md](PROTOCOLO_CAPTURA.md) respetando esa numeración.

## 4. Qué campos vienen de dónde

| Campo                                    | Origen previsto | Razón                                         |
| ---------------------------------------- | --------------- | --------------------------------------------- |
| `numero_serie`                           | **IA**          | Troquelado; es el identificador               |
| `marca`                                  | IA              | Troquelada                                    |
| `anio_fabricacion` / `fecha_fabricacion` | IA              | Troquelada                                    |
| `tipo_gas`                               | IA + backend    | Troquelado, pero el maestro manda             |
| `ph`                                     | IA              | Troquelada; cambia con cada prueba            |
| `procedencia`                            | IA              | Troquelada                                    |
| `peso`                                   | IA + backend    | Troquelada la tara; el maestro es más fiable  |
| `litros`                                 | IA + backend    | Ídem                                          |
| `unidad_medida`                          | IA + backend    | Ídem                                          |
| `color`                                  | IA + backend    | Visible, pero la luz altera el tono percibido |
| `m3`                                     | **Backend**     | No está troquelado                            |
| `ancho_diametro`                         | **Backend**     | No se mide en una foto sin referencia         |
| `largo_altura`                           | **Backend**     | Ídem                                          |

## 5. Estado y siguiente paso

Lo que **no** está bloqueado por la falta de fotos y ya está hecho:
microservicio completo, pipeline, API, catálogo, Docker, 78 pruebas, y los
scripts de dataset, entrenamiento, evaluación y calibración listos para
ejecutar.

Lo que **sí** está bloqueado: entrenar el detector, medir métricas reales y
calibrar los umbrales. Las tres cosas necesitan las fotografías.

**El siguiente paso del proyecto era la captura del dataset**, siguiendo
[PROTOCOLO_CAPTURA.md](PROTOCOLO_CAPTURA.md).

Ese paso ya está hecho: el dataset se importó el mismo día. El camino crítico
pasa ahora por revisar las anotaciones de caja que propone
`training/preannotate.py` y entrenar el detector.
