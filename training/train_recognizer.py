#!/usr/bin/env python3
"""Entrena un reconocedor especializado en números de serie troquelados.

Motivo: la evaluación de extremo a extremo dejó el diagnóstico claro. El
detector localiza el troquelado con confianza ~0.83, pero el OCR genérico lo
lee con un CER de 0.537, más de la mitad de los caracteres mal. Ese es el
techo del sistema, y no se sube ajustando umbrales.

Un reconocedor genérico está entrenado para leer miles de caracteres en
cualquier tipografía. Aquí hacen falta 21 símbolos en un estilo muy concreto,
y en la práctica son los diez dígitos más la 'K'. Un modelo pequeño
especializado tiene ventaja sobre uno general precisamente por eso.

Arquitectura: CRNN con pérdida CTC. Convoluciones que reducen la imagen a una
secuencia horizontal, una BiLSTM que la recorre y CTC para alinear sin
necesidad de anotar la posición de cada carácter. Es la arquitectura estándar
para texto en línea y, con este alfabeto, cabe en pocos megabytes y corre en
CPU.

Datos: los recortes reales extraídos por `build_ocr_dataset.py` más los
sintéticos de `synth_stamped.py`, que cubren los caracteres que apenas
aparecen en el inventario.

Uso:
    python training/train_recognizer.py --epochs 40
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ocr.charset import ALPHABET, BLANK_INDEX, NUM_CLASSES, decode, encode  # noqa: E402

# Los recortes se normalizan a una altura fija y un ancho máximo. 32 px de alto
# es lo habitual para reconocedores de línea y basta para estos caracteres.
IMG_HEIGHT = 32
IMG_WIDTH = 192


def prepare(image: np.ndarray) -> np.ndarray:
    """Normaliza un recorte: gris, altura fija y relleno por la derecha."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    scale = IMG_HEIGHT / max(height, 1)
    new_width = max(8, min(IMG_WIDTH, int(round(width * scale))))
    resized = cv2.resize(image, (new_width, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((IMG_HEIGHT, IMG_WIDTH), int(resized.mean()), dtype=np.uint8)
    canvas[:, :new_width] = resized

    # Normalización por muestra: el brillo absoluto varía muchísimo entre
    # fotografías y no aporta información sobre qué carácter es.
    array = canvas.astype(np.float32)
    array = (array - array.mean()) / (array.std() + 1e-6)
    return array


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Aumentos suaves. Nada de espejos: un serial reflejado no existe."""
    if rng.random() < 0.5:
        angle = rng.uniform(-3.5, 3.5)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=rng.uniform(0.2, 0.9))
    if rng.random() < 0.5:
        alpha = rng.uniform(0.75, 1.3)
        beta = rng.uniform(-25, 25)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return image


def load_pairs(listing: Path, root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    if not listing.exists():
        return pairs
    for line in listing.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        name, text = line.split("\t", 1)
        text = "".join(c for c in text.strip().upper() if c in ALPHABET)
        if text:
            pairs.append((root / name, text))
    return pairs


def build_model():
    from torch import nn

    class CRNN(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2, 2),                      # 16 x W/2
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2, 2),                      # 8 x W/4
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                # Sólo se reduce la altura: el eje horizontal es la secuencia
                # y comprimirlo perdería caracteres.
                nn.MaxPool2d((2, 1), (2, 1)),            # 4 x W/4
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d((4, 1), (4, 1)),            # 1 x W/4
            )
            self.rnn = nn.LSTM(128, 128, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=0.1)
            self.head = nn.Linear(256, num_classes)

        def forward(self, x):
            features = self.cnn(x)              # (B, 128, 1, W/4)
            features = features.squeeze(2).permute(0, 2, 1)   # (B, W/4, 128)
            sequence, _ = self.rnn(features)
            return self.head(sequence)          # (B, T, C)

    return CRNN(NUM_CLASSES)


def character_error_rate(expected: str, actual: str) -> float:
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for i, char_e in enumerate(expected, start=1):
        current = [i]
        for j, char_a in enumerate(actual, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_e != char_a)))
        previous = current
    return previous[-1] / len(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, default=Path("datasets/ocr"))
    parser.add_argument("--synth", type=Path, default=Path("datasets/ocr_synth"))
    parser.add_argument("--out", type=Path, default=Path("models/recognizer"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-oversample", type=int, default=16,
                        help="Veces que se repiten los recortes reales en el "
                             "entrenamiento, para compensar que son muchos "
                             "menos que los sintéticos.")
    parser.add_argument("--finetune-epochs", type=int, default=15,
                        help="Épocas finales sólo con fotografías reales.")
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERROR: falta torch. Instale: pip install -r requirements/train.txt")
        return 1

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    train_pairs = load_pairs(args.real / "train_label.txt", args.real)
    val_pairs = load_pairs(args.real / "val_label.txt", args.real)
    test_pairs = load_pairs(args.real / "test_label.txt", args.real)
    synth_pairs = load_pairs(args.synth / "label.txt", args.synth)

    if not train_pairs and not synth_pairs:
        print("ERROR: no hay datos. Ejecute build_ocr_dataset.py y synth_stamped.py")
        return 1

    # Los sintéticos sólo entran en entrenamiento. La validación y el test son
    # fotografías reales: es lo único que dice si el modelo sirve en planta.
    #
    # Los recortes reales se repiten varias veces porque son 243 frente a
    # 8000 sintéticos. Sin compensar esa proporción el modelo aprende muy bien
    # la textura sintética y luego falla en las fotografías: en una primera
    # prueba la pérdida bajó a 0.29 mientras el CER sobre imágenes reales se
    # quedaba en 0.65, peor que el OCR genérico. Es una brecha de dominio, no
    # falta de capacidad.
    combined = train_pairs * max(1, args.real_oversample) + synth_pairs
    print(f"entrenamiento: {len(combined)} muestras "
          f"({len(train_pairs)} reales x{args.real_oversample} + "
          f"{len(synth_pairs)} sintéticas)")
    print(f"validación: {len(val_pairs)} reales · test: {len(test_pairs)} reales")

    class Recognition(Dataset):
        def __init__(self, pairs, training: bool) -> None:
            self.pairs = pairs
            self.training = training

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index: int):
            path, text = self.pairs[index]
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            if self.training:
                image = augment(image, rng)
            array = prepare(image)
            return torch.from_numpy(array).unsqueeze(0), text

    def collate(batch):
        images = torch.stack([b[0] for b in batch])
        texts = [b[1] for b in batch]
        targets = torch.tensor([i for t in texts for i in encode(t)], dtype=torch.long)
        lengths = torch.tensor([len(encode(t)) for t in texts], dtype=torch.long)
        return images, targets, lengths, texts

    loader = DataLoader(Recognition(combined, True), batch_size=args.batch,
                        shuffle=True, collate_fn=collate, num_workers=0)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {device}")

    model = build_model().to(device)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")

    def evaluate(pairs) -> tuple[float, float]:
        model.eval()
        cers: list[float] = []
        exact = 0
        with torch.no_grad():
            for path, text in pairs:
                data = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                tensor = torch.from_numpy(prepare(image)).unsqueeze(0).unsqueeze(0)
                logits = model(tensor.to(device))
                indices = logits.argmax(dim=2)[0].tolist()
                predicted = decode(indices)
                cers.append(character_error_rate(text, predicted))
                exact += int(predicted == text)
        model.train()
        n = max(len(cers), 1)
        return sum(cers) / n, exact / n

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for images, targets, lengths, _ in loader:
            images = images.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), logits.size(1),
                                       dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()

        if epoch % 2 == 0 or epoch == args.epochs:
            cer, exact = evaluate(val_pairs or test_pairs)
            flag = ""
            if cer < best_cer:
                best_cer = cer
                torch.save(model.state_dict(), args.out / "recognizer.pt")
                flag = "  <- mejor"
            print(f"  época {epoch:>3}  pérdida={total / max(len(loader),1):.4f}  "
                  f"CER={cer:.3f}  exactos={exact:.1%}{flag}")

    # --- Fase 2: afinado sólo con fotografías reales ------------------------
    # Lo sintético sirve para que la red aprenda la forma de los caracteres;
    # lo real, para que aprenda cómo se ven de verdad. Terminar con unas
    # épocas sólo sobre fotografías, y con un paso de aprendizaje pequeño,
    # acerca el modelo al dominio en el que va a trabajar.
    if args.finetune_epochs > 0 and train_pairs:
        if (args.out / "recognizer.pt").exists():
            model.load_state_dict(
                torch.load(args.out / "recognizer.pt", map_location=device)
            )
        print(f"\nafinando {args.finetune_epochs} épocas sólo con "
              f"{len(train_pairs)} recortes reales")

        real_loader = DataLoader(
            Recognition(train_pairs * max(1, args.real_oversample // 2), True),
            batch_size=min(args.batch, 32), shuffle=True, collate_fn=collate,
        )
        fine_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 8,
                                           weight_decay=1e-4)
        for epoch in range(1, args.finetune_epochs + 1):
            total = 0.0
            for images, targets, lengths, _ in real_loader:
                images = images.to(device)
                logits = model(images)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), logits.size(1),
                                           dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, lengths)
                fine_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                fine_optimizer.step()
                total += float(loss.item())

            cer, exact = evaluate(val_pairs or test_pairs)
            flag = ""
            if cer < best_cer:
                best_cer = cer
                torch.save(model.state_dict(), args.out / "recognizer.pt")
                flag = "  <- mejor"
            print(f"  afinado {epoch:>3}  pérdida={total / max(len(real_loader),1):.4f}  "
                  f"CER={cer:.3f}  exactos={exact:.1%}{flag}")

    if (args.out / "recognizer.pt").exists():
        model.load_state_dict(torch.load(args.out / "recognizer.pt", map_location=device))

    print("\n" + "=" * 60)
    if test_pairs:
        cer, exact = evaluate(test_pairs)
        print(f"TEST (fotografías reales no vistas): CER={cer:.3f}  "
              f"seriales exactos={exact:.1%}")
        print("  referencia: el OCR genérico da CER=0.537 sobre el mismo dominio")
    print("=" * 60)

    # Exportación a ONNX: el servicio de inferencia no debe depender de torch.
    dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, device=device)
    onnx_path = args.out / "recognizer.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    (args.out / "alphabet.txt").write_text(ALPHABET, encoding="utf-8")
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\nModelo exportado: {onnx_path} ({size_mb:.1f} MB)")
    print("Para activarlo:  AI_SERIAL_RECOGNIZER_PATH=models/recognizer/recognizer.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
