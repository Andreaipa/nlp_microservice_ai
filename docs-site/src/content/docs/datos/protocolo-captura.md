---
title: Protocolo de captura
description: Cómo fotografiar los cilindros para que el sistema funcione.
---

> **Estado: la captura inicial está hecha.** El dataset importado cubre 100
> cilindros con las seis condiciones, una fotografía cada una (599 en total).
> Ver [DATASET.md](/datos/dataset/).
>
> Este documento sigue siendo la referencia para ampliar el dataset o
> reponerlo, y explica por qué cada condición importa. La sección de
> verificación al final se aplica igual a cualquier captura nueva.

El objetivo de una captura es que el sistema funcione en planta, no que las
fotos queden bonitas.

## Por qué importa el protocolo

El modelo aprende lo que se le enseña. Si todas las fotos se toman de frente,
a medio metro y con buena luz, el sistema funcionará exactamente en esas
condiciones y fallará el primer día que un operario fotografíe un cilindro
a contraluz en un rincón del almacén.

La variedad no es un adorno del dataset: es lo que determina si el sistema
sirve en planta.

## Qué se fotografía

El objetivo es el **collarín troquelado**, el anillo metálico de la parte
superior donde están grabados el número de serie y el resto de datos. No el
cilindro entero.

Conviene tener presente lo que hace difícil esta lectura:

- los caracteres están estampados en el metal, no impresos: carácter y fondo
  son del mismo color y sólo los distingue la sombra del relieve;
- la superficie es curva, de modo que el texto describe un arco;
- suele haber óxido, restos de pintura y suciedad encima;
- el metal refleja, y un reflejo directo sobre un carácter lo borra.

De ahí que el ángulo de la luz importe más que su cantidad.

## Estructura de carpetas

Una carpeta por cilindro, y dentro una por condición. Es la estructura que ya
existe en Drive y la que esperan los scripts:

```
datasets/raw/
├── Cilindro_001/
│   ├── 1.Frontal/
│   ├── 2.Diagonal/
│   ├── 3.Cerca/
│   ├── 4.Lejos/
│   ├── 5.Luz normal/
│   ├── 6.Poca luz/
│   └── Descripción/
│       └── Descripción.docx
├── Cilindro_002/
└── ...
```

El nombre de la carpeta de cilindro es el identificador que usan los scripts
para separar train/val/test. **Nunca mezcle fotos de dos cilindros en la misma
carpeta**: eso rompe la separación por identidad y falsea toda la evaluación.

## Las seis condiciones

| Carpeta        | Qué capturar                                                   | Mínimo |
| -------------- | -------------------------------------------------------------- | ------ |
| `1.Frontal`    | Collarín de frente, texto horizontal                           | 2      |
| `2.Diagonal`   | Desde unos 30-45° a izquierda y derecha                        | 3      |
| `3.Cerca`      | **El troquelado llenando el encuadre** (20-30 cm)              | 2      |
| `4.Lejos`      | Cilindro completo, como lo tomaría alguien con prisa (1,5-2 m) | 2      |
| `5.Luz normal` | Iluminación habitual de la planta                              | 3      |
| `6.Poca luz`   | Zona en sombra, sin flash                                      | 3      |

**Mínimo recomendado: 15 fotografías por cilindro.** La captura actual tiene
seis, una por condición, lo que ha resultado suficiente para arrancar pero deja
poco margen: cada fotografía adicional en las condiciones difíciles (frontal y
lejos, donde el troquelado apenas se distingue) mejora directamente el
resultado.

Si el tiempo aprieta, es mejor **15 fotos de 100 cilindros que 100 fotos de 15
cilindros**: lo que el modelo necesita aprender es a generalizar a cilindros
que no ha visto, y eso sólo lo aporta la variedad de cilindros.

> **Precisión importante sobre `3.Cerca`.** Al revisar las fotografías
> capturadas se vio que varias tomas "cercanas" encuadran el cilindro entero o
> se acercan a la válvula, por encima de la zona estampada. En esas imágenes no
> hay nada que leer. "Cerca" significa **cerca del troquelado**: el bloque de
> caracteres del hombro debe ocupar buena parte del encuadre, con el cilindro
> cortado por arriba y por abajo si hace falta. Es la condición con diferencia
> más productiva (60% de acierto frente a 13%), así que merece cuidado.

## Cómo tomar cada fotografía

1. Con el móvil que se usará en producción, o uno equivalente.
2. Enfocando el collarín, no el cuerpo del cilindro.
3. Sin zoom digital: acérquese físicamente.
4. Sin flash directo de frente: produce un reflejo que borra los caracteres.
   Si hace falta luz, ilumine **desde un lado**, en ángulo rasante: así el
   relieve proyecta sombra y el texto se lee mejor.
5. Limpie con un trapo sólo si el cilindro está muy sucio. Conviene conservar
   algunos cilindros sucios tal cual: el sistema los encontrará así.
6. No descarte fotos "malas" por estar algo movidas u oscuras. Son
   precisamente las que enseñan al sistema a reconocer cuándo debe pedir
   confirmación.

## Registro de los datos (imprescindible)

Por cada cilindro, rellene su `Descripción/Descripción.docx` con lo que está
troquelado, transcrito literalmente:

```
N° DE SERIE: 19S206055
MARCA: JP5
AÑO DE FABRICACIÓN: 09/2019
TIPO DE GAS: DIÓXIDO DE CARBONO CO2
UNIDAD DE MEDIDA: KG
PESO: 45.2 KG
ANCHO DIÁMETRO: 22 CM
LARGO ALTURA: 1.21 M
LITROS: 40.4 L
COLOR: GRIS
PH: 02/2025
PROCEDENCIA: CHN
```

Dos precisiones sobre este formato:

- **PH es la fecha de la Prueba Hidrostática** (mes/año), no una medida de
  acidez. Transcríbala tal como aparece.
- En la ficha del cilindro 002 figura `LARGO ALTURA: 1.21 CM`, que para un
  envase de 40 litros es imposible: son 1,21 m. El servicio detecta y corrige
  ese tipo de incoherencia, pero conviene anotar bien la unidad.

Si un campo no está troquelado en ese cilindro, **déjelo vacío**. No lo
complete con lo que "debería" poner: un dato inventado en la verdad de campo
corrompe la evaluación de forma silenciosa.

## Verificación antes de entrenar

Cuando termine la captura:

```bash
python training/audit_dataset.py --root datasets/raw
```

El informe dirá cuántos cilindros e imágenes hay, si alguna está corrupta o
duplicada, cómo se reparten las condiciones y si el conjunto es suficiente
para entrenar. Mientras devuelva bloqueantes, no tiene sentido pasar a anotar.