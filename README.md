# Explainable Campaign Targeting Simulator

Progetto per il corso **Data Science for Business**.

Il progetto sviluppa un sistema di uplift modeling per stimare l'effetto incrementale di una campagna marketing a livello di singolo cliente. Per ogni cliente vengono prodotti:

- probabilità stimata di acquisto con campagna;
- probabilità stimata di acquisto senza campagna;
- uplift score;
- raccomandazione operativa;
- cluster LTV;
- principali variabili rilevanti tramite SHAP.

Il modello finale è esposto tramite API REST con **FastAPI**, frontend web e container Docker.

---

## Descrizione

Non tutti i clienti reagiscono allo stesso modo a una campagna promozionale. Alcuni acquisterebbero comunque, altri non acquisterebbero in ogni caso, mentre una parte può essere influenzata dalla comunicazione.

L'obiettivo del progetto è stimare questo effetto incrementale, chiamato **uplift**, e usarlo per selezionare i clienti più adatti a ricevere la campagna.

L'approccio adottato è un **T-Learner**:

- un modello stima la probabilità di acquisto tra i clienti trattati;
- un secondo modello stima la probabilità di acquisto tra i clienti di controllo;
- la differenza tra le due probabilità produce lo `uplift_score`.

Oltre al modello uplift, il progetto include una proxy di **Customer Lifetime Value**, una segmentazione dei clienti in quattro fasce LTV, interpretabilità tramite SHAP e una simulazione di targeting uplift-based rispetto a una selezione casuale.

---

## Dataset

Dataset utilizzato: **RetailHero Uplift Modeling**, disponibile su HuggingFace come:

```text
pytorch-lifestream/retailhero-uplift
```

Il materiale del corso indicava il dataset `dllllb/retailhero-uplift`, ma questo namespace non risultava accessibile tramite gli endpoint HuggingFace utilizzati. La versione `pytorch-lifestream/retailhero-uplift` è stata quindi usata dopo verifica diretta di struttura, contenuto e coerenza delle tabelle.

Il dataset è composto da cinque tabelle:

| Tabella | Righe | Contenuto |
|---|---:|---|
| `clients` | 400.162 | Anagrafica cliente: età, genere, date di iscrizione/riscatto |
| `products` | 43.038 | Informazioni prodotto: categoria, brand, alcol, marca propria |
| `purchases` | 45.786.568 | Storico transazionale a livello di riga prodotto |
| `uplift_train` | 200.039 | Clienti con `treatment_flg` e `target` osservati |
| `uplift_test` | 200.123 | Clienti senza `target`, usati per la predizione finale |

L'audit iniziale del dataset viene eseguito tramite:

```bash
python src/data_audit.py
```

---

## Metodologia

Le scelte metodologiche principali sono riportate in dettaglio in `summary.ipynb`.

### Anti-leakage temporale

La tabella `purchases` copre il periodo dal 2018-11-21 al 2019-03-18. Poiché la finestra può includere comportamenti successivi alla campagna, le feature predittive sono calcolate solo sulle transazioni precedenti a:

```text
max(transaction_datetime) - 14 giorni
```

In questo modo si riduce il rischio che il modello utilizzi informazioni successive alla comunicazione.

### LTV proxy

La proxy di Customer Lifetime Value è calcolata come spesa storica complessiva del cliente.

La colonna `purchase_sum` è a livello di scontrino, ma compare su ogni riga prodotto appartenente alla stessa transazione. Per evitare una sovrastima della spesa, ogni `transaction_id` viene considerato una sola volta per ciascun cliente.

Il valore LTV è usato come indicatore di business e per la segmentazione, non come feature predittiva del modello uplift.

### Clustering LTV

I clienti vengono suddivisi in quattro fasce LTV tramite quartili:

- Basso
- Medio-basso
- Medio-alto
- Alto

I quartili sono stati preferiti a K-Means perché la distribuzione della spesa retail è fortemente asimmetrica e i quartili producono gruppi più interpretabili e bilanciati.

### Uplift modeling

Sono stati confrontati quattro approcci basati su T-Learner:

- Logistic Regression;
- Random Forest;
- XGBoost;
- AutoML tramite FLAML.

La metrica principale per la selezione del modello è il **Qini coefficient**, perché misura la qualità del ranking dei clienti in funzione dell'effetto incrementale stimato.

L'AUC dei singoli sotto-modelli viene comunque riportata, ma non è usata come metrica principale di selezione.

### Interpretabilità

L'interpretabilità è gestita tramite SHAP.

Poiché il T-Learner usa due modelli separati, il contributo SHAP all'uplift viene approssimato come differenza tra il contributo del modello treatment e quello del modello control.

---

## Struttura del repository

```text
retailhero-uplift-project/
├── README.md
├── summary.ipynb
├── requirements-audit.txt
├── .dockerignore
├── src/
│   ├── data_audit.py
│   ├── build_features.py
│   ├── build_modeling_dataset.py
│   ├── extract_ltv_thresholds.py
│   ├── train_uplift_models.py
│   ├── train_automl.py
│   ├── uplift_metrics.py
│   ├── explainability.py
│   ├── simulate_campaign.py
│   └── predict_test_set.py
├── models/
├── reports/
└── API_App/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        ├── main.py
        ├── schemas.py
        ├── model_utils.py
        ├── web_routes.py
        ├── templates/
        │   └── index.html
        └── api/
            └── routes.py
```

---

## Esecuzione della pipeline

Per riprodurre l'intera pipeline:

```bash
pip install -r requirements-audit.txt

python src/data_audit.py
python src/build_features.py
python src/build_modeling_dataset.py
python src/train_uplift_models.py
python src/train_automl.py
python src/extract_ltv_thresholds.py
python src/explainability.py
python src/simulate_campaign.py
python src/predict_test_set.py
```

Lo script `data_audit.py` scarica e salva in cache locale i dati grezzi nella cartella `data/raw/`. Gli script successivi lavorano sui file parquet locali.

---

## Esecuzione dell'applicazione

### Docker

```bash
cd API_App
docker build -t retailhero-uplift-api .
docker run -p 8000:8000 retailhero-uplift-api
```

Frontend:

```text
http://localhost:8000/
```

Documentazione API:

```text
http://localhost:8000/docs
```

### GitHub Container Registry

```bash
docker pull ghcr.io/mohitkabotra96/retailhero-uplift-api:latest
docker run -p 8000:8000 ghcr.io/mohitkabotra96/retailhero-uplift-api:latest
```

---

## Risultati principali

Il confronto tra modelli ha selezionato **Random Forest** come modello finale secondo il Qini coefficient.

| Modello | Qini coefficient |
|---|---:|
| Random Forest | 131.60 |
| Logistic Regression | 121.61 |
| AutoML / FLAML | 104.28 |
| XGBoost | 93.43 |

XGBoost ottiene l'AUC più alta sui singoli sotto-modelli, ma un Qini coefficient inferiore a quello di Random Forest. Questo indica che, nell'uplift modeling, una buona capacità predittiva sui singoli modelli non garantisce necessariamente un buon ranking dell'effetto incrementale.

Il progetto stima un uplift medio pari a circa 3 punti percentuali:

```text
Response rate trattati: 63,7%
Response rate controllo: 60,3%
```

La simulazione di campagna mostra che, con un budget pari al 20% dei clienti, la selezione uplift-based ottiene:

```text
323 conversioni incrementali stimate
```

contro:

```text
172 conversioni incrementali stimate con selezione casuale
```

Il miglioramento è pari a circa **+88%**.

---

## Output principali

Gli output generati dalla pipeline sono salvati in:

```text
models/
reports/
data/processed/
```

Output principali:

```text
models/uplift_random_forest_treatment.joblib
models/uplift_random_forest_control.joblib
models/feature_columns.json
models/ltv_thresholds.json

reports/shap_global_importance.csv
reports/shap_global_importance.png
reports/campaign_simulation.csv
reports/campaign_simulation.png
reports/test_set_predictions.csv

data/processed/customer_features.parquet
data/processed/customer_features_ltv.parquet
data/processed/train_modeling.parquet
data/processed/test_modeling.parquet
```

---

## Applicazione

L'applicazione FastAPI espone un endpoint principale:

```text
POST /predict
```

L'endpoint riceve le feature aggregate di un cliente e restituisce:

- `p_treatment`;
- `p_control`;
- `uplift_score`;
- raccomandazione operativa;
- `ltv_proxy`;
- cluster LTV;
- principali variabili rilevanti.

La raccomandazione è definita come:

```text
Includi se uplift_score > 0
Escludi altrimenti
```

---

## Autore

Mohit Kabotra

Progetto sviluppato per il corso **Data Science for Business** — Università degli Studi dell'Insubria.