.PHONY: help install install-train run test lint format import audit catalog split preannotate yolo-dataset pseudo-label train export evaluate calibrate openapi docs-install docs-sync docs-dev docs-build docker-build docker-up clean

# Origen del dataset para `make import`; sobrescríbalo si está en otra ruta.
SOURCE ?= $(HOME)/Downloads/Cilindros Electrametal

VENV := .venv
PY := $(VENV)/bin/python

help:
	@echo "Microservicio de IA para cilindros industriales"
	@echo ""
	@echo "  make install        Dependencias de inferencia + OCR"
	@echo "  make install-train  Añade la pila de entrenamiento (PyTorch)"
	@echo "  make run            Arranca el servicio en local"
	@echo "  make test           Ejecuta la suite de pruebas"
	@echo "  make lint           Comprueba estilo y tipos"
	@echo "  make format         Aplica formato (ruff + prettier) y sincroniza docs"
	@echo ""
	@echo "  make import         Importa el dataset normalizando nombres"
	@echo "  make audit          Audita el dataset antes de anotar"
	@echo "  make catalog        Construye catálogo y verdad de campo"
	@echo "  make split          Separa train/val/test por cilindro"
	@echo "  make preannotate    Propone anotaciones (revíselas después)"
	@echo "  make yolo-dataset   Ensambla la estructura para ultralytics"
	@echo "  make pseudo-label   Amplía anotaciones con el modelo entrenado"
	@echo "  make train          Entrena el detector"
	@echo "  make export         Exporta el detector a ONNX"
	@echo "  make evaluate       Mide el sistema de extremo a extremo"
	@echo "  make calibrate      Calibra los umbrales de confianza"
	@echo ""
	@echo "  make openapi        Exporta el esquema OpenAPI"
	@echo "  make docs-sync      Sincroniza docs/ con el sitio"
	@echo "  make docs-dev       Sitio de documentación en local"
	@echo "  make docs-build     Compila el sitio de documentación"
	@echo ""
	@echo "  make docker-build   Construye la imagen"
	@echo "  make docker-up      Levanta el servicio con docker compose"

$(VENV):
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

# RapidOCR va aparte y con --no-deps: ver requirements/ocr.txt.
install: $(VENV)
	$(PY) -m pip install -r requirements/dev.txt
	$(PY) -m pip install --no-deps -r requirements/ocr-engine.txt

install-train: $(VENV)
	$(PY) -m pip install -r requirements/train.txt

run:
	$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

# Python lo formatea ruff; el resto (Markdown, CSS, Astro, JSON) Prettier.
PRETTIER := ./docs-site/node_modules/.bin/prettier
PRETTIER_GLOB := "**/*.{js,mjs,ts,css,astro,md,mdx,json,yml,yaml}"

lint:
	$(VENV)/bin/ruff check app training tests
	$(VENV)/bin/mypy app --ignore-missing-imports
	@test -x $(PRETTIER) && $(PRETTIER) --check $(PRETTIER_GLOB) \
		|| echo "  (Prettier no instalado: ejecute make docs-install)"

format:
	$(VENV)/bin/ruff check app training tests --fix
	$(PRETTIER) --write $(PRETTIER_GLOB)
	$(PY) training/sync_docs.py

import:
	$(PY) training/import_dataset.py --source $(SOURCE)

audit:
	$(PY) training/audit_dataset.py --root datasets/raw --json models/dataset_audit.json

catalog:
	$(PY) training/build_catalog.py --root datasets/raw

preannotate:
	$(PY) training/preannotate.py --root datasets/raw

yolo-dataset:
	$(PY) training/prepare_yolo_dataset.py

pseudo-label:
	$(PY) training/pseudo_label.py --weights models/runs/detector/weights/best.pt \
		--only-split datasets/processed/train.txt

split:
	$(PY) training/split_by_cylinder.py --root datasets/raw --out datasets/processed

train:
	$(PY) training/train_detector.py --data datasets/yolo/data.yaml

export:
	$(PY) training/export_onnx.py --weights models/runs/detector/weights/best.pt

evaluate:
	$(PY) training/evaluate_pipeline.py

calibrate:
	$(PY) training/calibrate_thresholds.py

openapi:
	$(PY) training/export_openapi.py --out docs/openapi.json

docs-install:
	npm --prefix docs-site install

docs-sync:
	$(PY) training/sync_docs.py

docs-dev: openapi docs-sync
	npm --prefix docs-site run dev

docs-build: openapi docs-sync
	npm --prefix docs-site run build

docker-build:
	docker build -t ai-cylinder-service:1.0.0 .

docker-up:
	docker compose up -d --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
