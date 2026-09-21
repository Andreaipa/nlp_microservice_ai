#!/usr/bin/env python3
"""Generador de recortes sintéticos de troquelado.

El dataset real aporta 318 recortes, suficientes para afinar pero no para
cubrir bien el alfabeto: de los 20 caracteres que aparecen en los seriales del
inventario, seis (`X`, `T`, `G`, `H`, `L`, `P`) salen una o dos veces en total.
Un reconocedor entrenado sólo con eso no aprendería a distinguirlos.

Las muestras sintéticas rellenan ese hueco y además permiten forzar
condiciones que en el dataset están poco representadas: mucha sombra, óxido
abundante, desenfoque, texto muy inclinado.

Lo que se imita, tomado de observar los recortes reales:

* caracteres HUNDIDOS en el metal: el borde superior queda claro y el inferior
  en sombra, o al revés según de dónde venga la luz;
* fondo metálico con veteado, no plano;
* espaciado irregular entre caracteres, porque cada uno se estampa por
  separado con un punzón;
* pintura descascarada, óxido y suciedad encima;
* inclinación leve y curvatura, porque la superficie es cilíndrica.

Uso:
    python training/synth_stamped.py --count 4000
    python training/synth_stamped.py --count 200 --preview muestras.jpg
"""
from __future__ import annotations

import argparse
import glob
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Alfabeto real de los seriales del inventario.
ALPHABET = "0123456789EGHKLNPSTXY"

# Caracteres que apenas aparecen en los datos reales y que conviene
# sobrerrepresentar aquí para que el reconocedor llegue a verlos.
RARE = "XTGHLPEYNS"

FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Geneva.ttf",
]


def available_fonts() -> list[str]:
    fonts = [f for f in FONT_CANDIDATES if Path(f).exists()]
    if not fonts:
        fonts = sorted(glob.glob("/System/Library/Fonts/*.ttf"))[:6]
    if not fonts:  # pragma: no cover - entorno sin fuentes del sistema
        fonts = sorted(glob.glob("/usr/share/fonts/**/*.ttf", recursive=True))[:6]
    return fonts


def random_serial(rng: random.Random) -> str:
    """Genera un serial con la forma de los del inventario."""
    shape = rng.choices(
        ["DDADDDDDD", "ADDDDDDD", "DDDDDD", "ADDDDDD", "DDDDD", "DDDDDDDA",
         "DDDDDDDDD", "ADDDDD", "AAADDDD"],
        weights=[42, 34, 7, 6, 4, 3, 1, 1, 1],
    )[0]
    out = []
    for kind in shape:
        if kind == "D":
            out.append(rng.choice(string.digits))
        else:
            # Se inclina la balanza hacia los caracteres poco vistos.
            out.append(rng.choice(RARE if rng.random() < 0.6 else ALPHABET))
    return "".join(out)


def metal_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    """Fondo metálico con veteado, manchas de pintura y óxido."""
    base = rng.randint(70, 190)
    canvas = np.full((height, width), base, dtype=np.float32)

    # Veteado: gradiente suave más ruido de baja frecuencia.
    gradient = np.linspace(-rng.randint(5, 30), rng.randint(5, 30), width)
    canvas += gradient[None, :]
    noise = rng.randint(4, 15) * np.random.default_rng(rng.randint(0, 10**6)).standard_normal(
        (max(height // 8, 2), max(width // 8, 2))
    )
    canvas += cv2.resize(noise.astype(np.float32), (width, height),
                         interpolation=cv2.INTER_CUBIC)

    # Manchas: pintura descascarada u óxido.
    for _ in range(rng.randint(0, 5)):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        axes = (rng.randint(width // 12 + 2, width // 3),
                rng.randint(height // 6 + 1, height))
        patch = np.zeros_like(canvas)
        cv2.ellipse(patch, (cx, cy), axes, rng.randint(0, 180), 0, 360,
                    float(rng.randint(-32, 32)), -1)
        canvas += cv2.GaussianBlur(patch, (0, 0), sigmaX=rng.uniform(2, 9))

    return np.clip(canvas, 0, 255)


def render_text_mask(text: str, width: int, height: int, font_path: str,
                     rng: random.Random) -> np.ndarray:
    """Dibuja el texto con espaciado irregular, carácter a carácter."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    size = int(height * rng.uniform(0.52, 0.78))
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:  # pragma: no cover - fuente ilegible
        font = ImageFont.load_default()

    # Ancho total aproximado para centrar.
    widths = []
    for char in text:
        box = draw.textbbox((0, 0), char, font=font)
        widths.append(box[2] - box[0])
    spacing = int(size * rng.uniform(0.10, 0.32))
    total = sum(widths) + spacing * (len(text) - 1)

    x = (width - total) / 2
    baseline = (height - size) / 2
    for char, char_width in zip(text, widths, strict=True):
        # Cada punzón cae ligeramente desalineado respecto al anterior.
        dy = rng.uniform(-size * 0.07, size * 0.07)
        draw.text((x, baseline + dy), char, fill=255, font=font)
        x += char_width + spacing * rng.uniform(0.7, 1.3)

    return np.asarray(mask, dtype=np.float32) / 255.0


def emboss(background: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Convierte la máscara de texto en relieve sobre el fondo.

    Un carácter troquelado no cambia de color: hunde el metal. Lo que se ve es
    el borde iluminado por un lado y la sombra por el otro, y de qué lado cae
    cada cosa depende de por dónde entre la luz.
    """
    shift = rng.randint(1, 4)
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = int(round(np.cos(angle) * shift)), int(round(np.sin(angle) * shift))

    highlight = np.roll(np.roll(mask, -dy, axis=0), -dx, axis=1)
    shadow = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)

    # El rango se fijó comparando con los recortes reales: por debajo de 25 el
    # texto se confunde con el veteado del metal y la muestra deja de ser
    # legible incluso para una persona. Entrenar con muestras ilegibles enseña
    # al modelo a adivinar, que es justo lo que no se quiere.
    strength = rng.uniform(28, 75)
    result = background + highlight * strength - shadow * strength

    # El fondo del hueco queda algo más oscuro que la superficie.
    result -= mask * rng.uniform(0, 12)
    return result


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Añade los defectos de una fotografía tomada a pulso en un almacén."""
    if rng.random() < 0.6:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=rng.uniform(0.3, 1.0))

    if rng.random() < 0.5:
        height, width = image.shape[:2]
        angle = rng.uniform(-4.0, 4.0)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    image = image + rng.uniform(1, 5) * np.random.default_rng(
        rng.randint(0, 10**6)
    ).standard_normal(image.shape)

    if rng.random() < 0.18:
        # Reflejo especular: una franja quemada que borra parte del texto.
        height, width = image.shape[:2]
        glare = np.zeros_like(image)
        cv2.line(glare, (rng.randint(0, width), 0), (rng.randint(0, width), height),
                 float(rng.randint(20, 55)), rng.randint(5, 18))
        image = image + cv2.GaussianBlur(glare, (0, 0), sigmaX=rng.uniform(4, 14))

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_sample(text: str, font_path: str, rng: random.Random) -> np.ndarray:
    height = rng.randint(28, 64)
    width = int(height * rng.uniform(3.2, 6.5) * max(len(text), 1) / 8)
    width = max(width, 64)

    background = metal_background(width, height, rng)
    mask = render_text_mask(text, width, height, font_path, rng)
    embossed = emboss(background, mask, rng)
    return degrade(embossed, rng)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=Path("datasets/ocr_synth"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=Path, default=None,
                        help="Guarda un mosaico de muestras para inspección.")
    args = parser.parse_args()

    fonts = available_fonts()
    if not fonts:
        print("ERROR: no se encontró ninguna fuente TrueType en el sistema.")
        return 1

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    previews: list[np.ndarray] = []

    for index in range(args.count):
        text = random_serial(rng)
        sample = generate_sample(text, rng.choice(fonts), rng)
        name = f"synth_{index:06d}.jpg"
        cv2.imwrite(str(args.out / name), sample)
        rows.append(f"{name}\t{text}")
        if args.preview and len(previews) < 12:
            previews.append(cv2.resize(sample, (320, 64)))

    (args.out / "label.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if args.preview and previews:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview),
                    np.vstack([np.hstack(previews[i:i + 2])
                               for i in range(0, len(previews) - 1, 2)]))

    print(f"Generadas {args.count} muestras en {args.out}")
    print(f"Fuentes utilizadas: {len(fonts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
