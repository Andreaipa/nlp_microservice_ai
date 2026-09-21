---
title: El dataset
description: Contenido real, hallazgos e inconsistencias del inventario.
---

Importado el 2026-09-20 desde `~/Downloads/Cilindros Electrametal` mediante
`training/import_dataset.py`.

## Qué hay

|                       |                      |
| --------------------- | -------------------- |
| Cilindros             | 100                  |
| Fotografías           | 599                  |
| Fichas de descripción | 100 `.docx`          |
| Imágenes corruptas    | 0                    |
| Resolución dominante  | 2296 × 4080 (9,4 MP) |
| Tamaño                | 1,5 GB               |

Cobertura por condición de captura, prácticamente perfecta:

| Condición    | Imágenes |
| ------------ | -------- |
| 1.Frontal    | 100      |
| 2.Diagonal   | 100      |
| 3.Cerca      | 100      |
| 4.Lejos      | 99       |
| 5.Luz normal | 100      |
| 6.Poca luz   | 100      |

Es **una fotografía por condición y cilindro**, no las "100 por cilindro" que
se mencionaron al principio. Para entrenar un detector de regiones es
suficiente: lo que importa es la variedad de cilindros, y hay 100 distintos.

Calidad medida sobre el conjunto: nitidez mediana 271 (imágenes nítidas),
brillo medio 79 sobre 255 (oscuras, coherente con un almacén) y contraste
medio 64. Sólo 3 imágenes (0,5%) quedan por debajo de los umbrales de calidad.

## Qué se ve en las fotos

El troquelado **no está en el collarín** sino en el **hombro** del cilindro, la
zona cónica bajo la válvula. Aparece grabado sobre pintura gris descascarada,
con óxido encima y, en muchos casos, etiquetas adhesivas alrededor.

Esto confirma que la detección de región es imprescindible y no un adorno: en
las tomas `1.Frontal` y `4.Lejos` el troquelado ocupa una fracción minúscula
de una imagen de 9 MP, y además suele haber otros cilindros al fondo. Pasar la
imagen completa al OCR no funciona.

## Hallazgos en las fichas

Las 100 fichas están escritas a mano y eso se nota. `training/build_catalog.py`
normaliza lo que puede e informa del resto.

### El campo PH tiene dos significados distintos

| Contenido                                    | Cilindros |
| -------------------------------------------- | --------- |
| Periodicidad (`CADA 10 AÑOS`, `CADA 5 AÑOS`) | 64        |
| Fecha concreta (`07/2026`, `02/2025`)        | 25        |
| `SN` o vacío                                 | 8         |

Son dos datos diferentes y se modelan por separado: `ph` (fecha AAAA-MM) y
`ph_periodicidad_anios` (entero). De "cada 10 años" no se deduce cuándo fue la
última prueba, así que afirmar una fecha ahí sería inventar.

### Un serial duplicado

`K5738166` aparece en **Cilindro_051 y Cilindro_053**.

Es un problema de datos, no del sistema: un identificador repetido deja de
identificar. O es un error de transcripción en una de las dos fichas, o hay
dos cilindros marcados igual. Conviene resolverlo con la empresa, porque
ninguna tecnología puede distinguir dos activos con el mismo número.

### Los seriales no comparten formato

Doce patrones distintos y longitudes de 3 a 10 caracteres:

| Patrón                        | Cilindros |
| ----------------------------- | --------- |
| `DDADDDDDD` (ej. `19S206055`) | 41        |
| `ADDDDDDD` (ej. `K5738028`)   | 34        |
| `DDDDDD`                      | 6         |
| `ADDDDDD`                     | 5         |
| `DDDDD`                       | 4         |
| otros 7 patrones              | 9         |

Esto valida la decisión de no fijar una máscara posicional escrita a mano: la
inferencia sobre el catálogo detecta correctamente que no hay estructura común
y se limita a los márgenes de longitud, sin corregir caracteres a ciegas.

### Doce seriales demasiado cortos, y por qué importa

El inventario contiene doce cilindros con seriales de seis caracteres o menos:

| Serial   | Cilindro |     | Serial   | Cilindro |
| -------- | -------- | --- | -------- | -------- |
| `804`    | 081      |     | `001087` | 057      |
| `20640`  | 015      |     | `001029` | 058      |
| `30692`  | 040      |     | `612825` | 077      |
| `42705`  | 065      |     | `673165` | 078      |
| `85343`  | 066      |     | `610782` | 080      |
| `P30176` | 083      |     | `610747` | 082      |

No es un detalle cosmético. Al medir el sistema sobre el conjunto de test
apareció este caso: en una fotografía del cilindro **`21S062189`** el OCR leyó
el fragmento **`20640`**, que resulta ser el serial **completo** del
**Cilindro_015**. El sistema lo habría dado por identificado y el movimiento
habría quedado registrado en el cilindro equivocado, sin que nada lo delatara.

Es el fallo más dañino que puede tener un sistema de trazabilidad, porque no
se nota: el inventario se descuadra en silencio.

La salvaguarda aplicada es marcar todos esos seriales para confirmación
manual, incluso cuando la lectura es perfecta y los umbrales están calibrados.
Son doce cilindros de cien que siempre pedirán una confirmación. La
alternativa —aceptarlos automáticamente— es asumir que cualquier número corto
leído en una foto puede reasignar un activo.

**Recomendación para la empresa:** remarcar esos doce cilindros con un código
más largo elimina el problema de raíz y recupera la identificación automática
para ellos.

### Otras inconsistencias normalizadas

- **Altura en unidades equivocadas**: casi todas las fichas escriben
  `1.48 CM` queriendo decir 1,48 m. Se convierte y se deja constancia.
- **Procedencia en dos formatos**: `CHN` (47) y `CN` (35), más nombres
  completos (`ARGENTINA`, `BRAZIL`, `BRASIL`). Todo se unifica a ISO-3.
- **`SN` como marcador de dato ausente** en marca, PH y procedencia. No es un
  valor: se trata como campo vacío.
- **Mezclas de gas**: `80% ARGON 20% DIOXIDO DE CARBONO` en 7 cilindros. Una
  búsqueda ingenua por subcadena lo clasificaría como argón puro; se detecta
  como mezcla y se codifica `MIX`.
- **Erratas de color**: `MAARON` por `MARRON`, y compuestos como
  `MARRON/GRIS`, que se conservan porque un cilindro bicolor es información
  real.

### Cobertura de campos tras normalizar

| Campo                                            | Cobertura |
| ------------------------------------------------ | --------- |
| numero_serie, unidad_medida, largo_altura, color | 100%      |
| tipo_gas                                         | 99%       |
| procedencia                                      | 98%       |
| marca                                            | 97%       |
| peso                                             | 94%       |
| fecha_fabricacion / anio_fabricacion             | 89%       |
| litros                                           | 86%       |
| ancho_diametro                                   | 83%       |
| ph_periodicidad_anios                            | 66%       |
| ph (fecha)                                       | 25%       |

Distribuciones: gas mayoritariamente oxígeno (46) y argón (19); marcas JP (41),
JD (34) y NORRIS (9); color verde (48) y marrón (19); unidad M3 (83) sobre KG (16).

## Reparto train / val / test

Hecho por identidad de cilindro con `training/split_by_cylinder.py`:

| Subconjunto | Cilindros | Imágenes |
| ----------- | --------- | -------- |
| train       | 70        | 419      |
| val         | 15        | 90       |
| test        | 15        | 90       |

Ningún cilindro aparece en más de un subconjunto, así que el test mide
generalización a cilindros nunca vistos, que es exactamente lo que hará el
sistema en planta.

## Reproducir

```bash
python training/import_dataset.py --source ~/Downloads/"Cilindros Electrametal"
python training/audit_dataset.py --root datasets/raw
python training/build_catalog.py --root datasets/raw
python training/split_by_cylinder.py --root datasets/raw --out datasets/processed
```

## Limitaciones observadas al revisar casos reales

Al generar la evidencia visual (`training/demo_report.py`) sobre tomas
cercanas del conjunto de test aparecieron dos problemas que las métricas
agregadas no muestran:

### El OCR prefiere las etiquetas adhesivas al troquelado

En el Cilindro_034 el número de serie `171392042` está perfectamente visible
en la fotografía, y sin embargo el sistema leyó `201707`, que es la fecha
estampada justo debajo. El OCR había reconocido además `TRUJILLO`,
`Argonindustal` y `161048355`, todo ello texto **impreso** de las pegatinas
que rodean al collarín.

Es comprensible: el texto impreso tiene mucho más contraste que el troquelado,
así que el reconocedor lo encuentra primero. El remedio no es tocar el OCR
sino mejorar la clase `serial_text`, que acota la línea concreta del número y
hoy tiene un mAP de sólo 0.155. Cuando esa detección funcione, el recorte que
llega al OCR no contendrá etiquetas.

### "Cerca" no siempre significa cerca del troquelado

Dos de las seis tomas `3.Cerca` revisadas no muestran el collarín: una encuadra
el cilindro entero y otra se acerca a la válvula, por encima de la zona
estampada. En esas fotografías no hay nada que leer, por buena que sea la
lectura.

Para la próxima captura conviene precisar la instrucción: **cerca del
troquelado**, no cerca del cilindro. El protocolo actualizado está en
[PROTOCOLO_CAPTURA.md](/datos/protocolo-captura/).