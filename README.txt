# PredDist Starter (Flask local)

## Requisitos
- Python 3.10+
- Pip y venv
- (Opcional) Excel para abrir archivos .xlsx

## Pasos rápidos (Windows PowerShell)
```ps1
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
flask --app app.py init-db
python app.py
```
Abre http://127.0.0.1:5000  (usuario demo: admin / admin)

## Pasos rápidos (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
flask --app app.py init-db
python app.py
```

## Empaquetar a .exe (Windows)
```ps1
pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" app.py
# Ejecutable en dist/app.exe
```

## Estructura del CSV/Excel
Encabezados: sku, product_name, store, quantity, date (YYYY-MM-DD)
