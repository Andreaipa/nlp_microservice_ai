---
title: Estrategia de anotación
description: Clases, criterios y pre-anotación automática.
---

## Principio

Anotar es el trabajo más caro del proyecto y el que más condiciona el
resultado. La regla que gobierna las decisiones de abajo: **una clase existe
sólo si implica un recorte distinto o un tratamiento distinto aguas abajo**.
Todo lo que el OCR y el parser ya separan dentro de una región no necesita
clase propia.

## Clases

| Clase          | Qué encierra                      | Para qué sirve                              |
| -------------- | --------------------------------- | ------------------------------------------- |
| `cylinder`     | El envase completo                | Encuadre, descartar fondo, contar cilindros |
| `marking_area` | El collarín troquelado entero     | Es el recorte que va al OCR                 |
| `serial_text`  | Sólo la línea del número de serie | Recorte prioritario, el dato crítico        |

Tres clases, no doce. Crear `marca_text`, `peso_text`, `gas_text` y demás
multiplicaría el coste de anotación y el riesgo de confusión entre regiones
visualmente idénticas, sin aportar nada: el parser ya distingue "45.2 KG" de
"09/2019" por su forma.

**Esta lista es una propuesta, no una conclusión.** Debe revisarse al ver las
primeras fotos reales:

- si el troquelado aparece siempre como un bloque compacto, `serial_text`
  puede sobrar y bastaría `marking_area`;
- si hay cilindros con etiqueta adhesiva impresa además del troquelado, hará
  falta una clase para ella, porque el preprocesado que necesita es distinto
  (texto impreso, no relieve);
- si aparecen collarines con el troquelado repetido en dos zonas, conviene
  anotarlas ambas.

## Pre-anotación automática

Antes de dibujar nada a mano, conviene ejecutar:

```bash
python training/preannotate.py --root datasets/raw --workers 6
```

Lo que hace es aprovechar algo que ya tenemos: **sabemos qué número de serie
lleva cada fotografía**, porque está en la ficha del cilindro. Así que no hay
que adivinar dónde está el troquelado. El script pasa el OCR por la franja
superior de la imagen y se queda con la región cuyo texto coincide con el
serial esperado; de ahí derivan las cajas de `serial_text` y `marking_area`.

El criterio de aceptación es deliberadamente estricto, y lo es por una razón
concreta. La primera versión sólo exigía parecido de texto, y al revisar las
propuestas aparecieron cajas sobre el cuerpo del cilindro, en zonas de pintura
descascarada donde **no hay texto alguno**: el OCR llegaba a devolver una
cadena parecida al serial con 0.96 de confianza sobre pura textura. Ahora se
exige además que la caja tenga forma de línea horizontal y esté en la parte
alta del envase, que es donde está el troquelado.

Una caja mal puesta es peor que ninguna: enseña al detector a mirar donde no
debe. Por eso se prefiere anotar menos y bien.

**Cobertura real sobre este dataset: 54% (326 de 599 imágenes).**

| Condición    | Cobertura |
| ------------ | --------- |
| 3.Cerca      | 73%       |
| 1.Frontal    | 53%       |
| 2.Diagonal   | 53%       |
| 6.Poca luz   | 53%       |
| 5.Luz normal | 50%       |
| 4.Lejos      | 44%       |

Llegar aquí costó tres correcciones que conviene no deshacer:

1. **Motor OCR.** Con PaddleOCR cada imagen costaba más de un minuto, lo que
   hacía inviable explorar. Con RapidOCR bajó a unos segundos y permitió todo
   lo demás.
2. **Búsqueda multi-escala.** Repetir la búsqueda sobre la imagen ampliada al
   doble recupera el troquelado diminuto de las tomas lejanas. Subió la
   cobertura del 23% al 30%.
3. **Un error en el filtro de posición vertical.** La búsqueda se hace sobre
   la banda superior de la imagen, y la posición del hallazgo se comparaba
   contra la banda en lugar de contra la imagen completa. Como la banda ya era
   el 60% superior, el filtro exigía en realidad el 36% superior y descartaba
   lecturas correctas. Se notó porque `3.Cerca`, que debería ser la condición
   más fácil, daba un 4% mientras el resto rondaba el 50%. Corregido, pasó al
   **73%**.

El tercero es el más instructivo: el OCR leía el serial correctamente, con
0.92 de confianza y la forma esperada, y era el código propio el que lo
tiraba. No se ve revisando el código, sólo comparando resultados por
condición contra la verdad de campo.

## Pseudo-etiquetado: cerrar el círculo

Una vez entrenado el primer detector, ya sabe reconocer el hombro **también
donde el texto no se lee**, que es justo lo que el OCR no podía resolver:

```bash
python training/pseudo_label.py \
    --weights models/runs/detector/weights/best.pt \
    --only-split datasets/processed/train.txt \
    --conf 0.70
```

Sus propuestas se escriben en `datasets/labels_pseudo/`, aparte de las
fiables, para revisarlas antes de mezclarlas. Después se reentrena y se repite.
Cada vuelta amplía la cobertura.

Dos reglas que no conviene saltarse:

- **Nunca pseudo-etiquetar el conjunto de test.** Las cajas vendrían del propio
  modelo que se va a evaluar, y la métrica dejaría de medir nada. De ahí
  `--only-split train.txt`.
- **Umbral de confianza alto** (0.70 por omisión). En pseudo-etiquetado un
  error no se queda quieto: se convierte en ejemplo de entrenamiento y se
  refuerza en la vuelta siguiente.

## Cómo anotar (y revisar) a mano

Herramienta sugerida: **Label Studio** o **CVAT**, ambas importan y exportan
formato YOLO, así que pueden cargar las propuestas de `datasets/labels/` para
revisarlas en lugar de empezar de cero. Roboflow también sirve si se acepta
subir las imágenes a un servicio externo.

Hay que hacer tres cosas sobre las propuestas:

1. **Verificar** que cada caja cae sobre el troquelado. Descartar las que no.
2. **Completar** las imágenes que el script no pudo resolver.
3. **Añadir la clase `cylinder`**, que no se pre-anota: delimitar el envase
   completo no se deduce del texto y requiere criterio humano.

Criterios:

1. `marking_area` debe incluir **todo** el troquelado con un margen pequeño.
   Si la caja corta un carácter, el OCR leerá mal ese carácter siempre.
2. `serial_text` se ciñe a la línea del número de serie, sin las líneas de
   alrededor.
3. Si el collarín está parcialmente tapado o girado, anótelo igual: el modelo
   tiene que aprender a encontrarlo en esas condiciones.
4. Si el troquelado es ilegible incluso para una persona, **no lo anote como
   `serial_text`**; anote sólo `marking_area`. Enseñar al modelo a señalar
   como serial algo que nadie puede leer produce falsos positivos seguros.
5. Sea consistente en los márgenes entre anotadores. Una diferencia
   sistemática de criterio se traduce en cajas peores.

## Formato YOLO

Un `.txt` por imagen, misma ruta y nombre:

```
<clase> <x_centro> <y_centro> <ancho> <alto>
```

Todo normalizado entre 0 y 1. Índices de clase: `0=cylinder`,
`1=marking_area`, `2=serial_text`, en ese orden, que es el que exporta
`training/export_onnx.py` en `detector.classes.json`.

## data.yaml

```yaml
path: /ruta/absoluta/a/datasets
train: processed/train.txt
val: processed/val.txt
test: processed/test.txt

names:
  0: cylinder
  1: marking_area
  2: serial_text
```

Los listados los genera `training/split_by_cylinder.py`, que reparte por
identidad de cilindro. **No los escriba a mano**: un reparto aleatorio de
imágenes mete fotos del mismo cilindro en train y test a la vez, y a partir de
ahí todas las métricas mienten.

## Verdad de campo para la evaluación

Además de las cajas, hace falta un CSV que asocie cada imagen con su serial
real, para medir el sistema completo:

```csv
imagen,numero_serie
Cilindro_002/5.Luz normal/IMG_001.jpg,19S206055
Cilindro_002/6.Poca luz/IMG_007.jpg,19S206055
```

Se genera a partir de los `Descripción.docx` de cada cilindro. Si un serial se
transcribe mal aquí, el sistema parecerá fallar cuando en realidad acierta, y
la calibración de umbrales saldrá desviada.

## Orden de trabajo recomendado

1. Ejecutar la pre-anotación sobre todo el dataset.
2. Revisar las propuestas de **20 cilindros**.
3. Entrenar con eso y mirar las métricas.
4. Revisar en qué falla y **sólo entonces** decidir si las clases son las
   adecuadas.
5. Completar el resto con los criterios ya asentados.

Revisar los 100 cilindros antes de haber entrenado una sola vez arriesga
repetir 100 veces un criterio equivocado.

## Ensamblar el dataset para entrenar

```bash
python training/prepare_yolo_dataset.py
```

Construye `datasets/yolo/` con la estructura que espera ultralytics
(`images/` y `labels/` paralelos), enlazando las imágenes en lugar de
copiarlas para no duplicar 1,5 GB. El reparto lo toma del manifiesto de
`split_by_cylinder.py`, así que no puede divergir de él.

Las imágenes sin etiqueta quedan fuera a propósito: para el detector, una
imagen sin cajas significa "aquí no hay nada que detectar", justo lo
contrario de lo que se quiere enseñar.

**Sobre el rebalanceo de validación.** El reparto de `split_by_cylinder.py`
equilibra las 599 fotografías, pero sólo una fracción llega a estar anotada, y
esa fracción puede quedar muy desigual: en la primera pasada la validación se
quedó con tres imágenes, insuficientes para decidir una parada temprana. El
script traslada entonces cilindros **completos** de entrenamiento a
validación hasta alcanzar `--min-val` (10 por omisión).

Se mueven cilindros enteros y nunca fotografías sueltas, porque partir un
cilindro entre dos subconjuntos reintroduce exactamente la fuga de datos que
el reparto por identidad evita. El conjunto de test no se toca en ningún
caso: es la referencia de la evaluación final.