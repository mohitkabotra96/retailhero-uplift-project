# Explainable Campaign Targeting Simulator - API_App
#
# Immagine leggera che serve backend (API /predict) + frontend (form web),
# usando modelli GIA' ALLENATI (nessun training in questo container -- vedi
# nota metodologica in README.md sulla separazione training/serving).
#
# Build context atteso: questa stessa cartella (quella che contiene
# Dockerfile, requirements.txt, app/, models/). Da lanciare cosi':
#
#   docker build -t retailhero-uplift-api .
#
# (eseguito da DENTRO questa cartella, non dalla cartella superiore)

FROM python:3.11-slim

WORKDIR /app

# Installa le dipendenze Python prima di copiare il codice, per
# sfruttare la cache Docker sui layer (requirements.txt cambia raramente)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codice applicativo
COPY app/ ./app/

# Modelli gia' allenati
COPY models/ ./models/

WORKDIR /app/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
