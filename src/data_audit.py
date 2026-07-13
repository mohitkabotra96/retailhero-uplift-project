from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd



DATASET_ID = "pytorch-lifestream/retailhero-uplift"
CONFIGS = ["clients", "products", "purchases", "uplift_train", "uplift_test"]
SPLIT = "train" 

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = Path(__file__).resolve().parent.parent / "data_audit.md"


def get_parquet_urls(config: str) -> list[str]:
    """Recupera gli URL parquet per un config tramite l'endpoint HF ufficiale."""
    import requests

    api_url = f"https://huggingface.co/api/datasets/{DATASET_ID}/parquet/{config}/{SPLIT}"
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    files = resp.json()

    if not files:
        raise RuntimeError(
            f"Nessun file parquet trovato per config='{config}'. "
            f"Controllare manualmente: {api_url}"
        )

    urls = []
    for f in files:
        if f.startswith("http"):
            urls.append(f)
        else:
            urls.append(
                f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{f}"
            )
    return urls


def load_config(config: str) -> pd.DataFrame:
    """Scarica (con cache locale) e carica un config come DataFrame."""
    cache_path = RAW_DIR / f"{config}.parquet"

    if cache_path.exists():
        print(f"[{config}] uso cache locale: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[{config}] recupero URL parquet...")
    urls = get_parquet_urls(config)
    print(f"[{config}] trovati {len(urls)} file parquet, scarico...")

    dfs = [pd.read_parquet(u) for u in urls]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    df.to_parquet(cache_path)
    print(f"[{config}] salvato in cache: {cache_path} ({len(df):,} righe)")
    return df


def profile_dataframe(name: str, df: pd.DataFrame) -> str:
    """Genera un blocco markdown con il profilo di un DataFrame."""
    lines = [f"## Config: `{name}`", ""]
    lines.append(f"- Shape: **{df.shape[0]:,} righe × {df.shape[1]} colonne**")
    lines.append("")

    lines.append("### Colonne e tipi")
    lines.append("")
    lines.append("| colonna | dtype | % null | n unique |")
    lines.append("|---|---|---|---|")
    for col in df.columns:
        n_null_pct = df[col].isna().mean() * 100
        n_unique = df[col].nunique(dropna=True)
        lines.append(f"| {col} | {df[col].dtype} | {n_null_pct:.2f}% | {n_unique:,} |")
    lines.append("")

    lines.append("### Prime righe")
    lines.append("")
    lines.append("```")
    lines.append(df.head(5).to_string())
    lines.append("```")
    lines.append("")


    low_card_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 10]
    if low_card_cols:
        lines.append("### Value counts (colonne a bassa cardinalità, utile per target/treatment/flag)")
        lines.append("")
        for col in low_card_cols:
            lines.append(f"**{col}**")
            lines.append("```")
            lines.append(df[col].value_counts(dropna=False).to_string())
            lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    report_sections = [
        "# Data Audit — pytorch-lifestream/retailhero-uplift",
        "",
        "Report generato automaticamente da `src/data_audit.py`.",
        "Questo file è il riferimento usato nel progetto per verificare colonne, tipi e dimensioni reali del dataset.",
        "Qualunque scelta di feature engineering, target, treatment o proxy LTV",
        "nel resto del progetto fa riferimento a quanto riportato qui,",
        "non a ipotesi.",
        "",
    ]

    dfs: dict[str, pd.DataFrame] = {}

    for config in CONFIGS:
        print(f"\n{'=' * 60}\nConfig: {config}\n{'=' * 60}")
        try:
            df = load_config(config)
            dfs[config] = df
            section = profile_dataframe(config, df)
            report_sections.append(section)
        except Exception as e:  
            err_msg = f"## Config: `{config}`\n\n**ERRORE nel caricamento**: {e}\n"
            print(f"[{config}] ERRORE: {e}", file=sys.stderr)
            report_sections.append(err_msg)


    report_sections.append("## Controlli di coerenza tra config")
    report_sections.append("")

    if "clients" in dfs and "purchases" in dfs:
        clients_ids = set(dfs["clients"]["client_id"])
        purchases_ids = set(dfs["purchases"]["client_id"]) if "client_id" in dfs["purchases"].columns else set()
        overlap = len(clients_ids & purchases_ids)
        report_sections.append(
            f"- `clients.client_id` ∩ `purchases.client_id`: {overlap:,} "
            f"(clients totali: {len(clients_ids):,}, purchases totali: {len(purchases_ids):,})"
        )

    if "uplift_train" in dfs and "clients" in dfs:
        train_ids = set(dfs["uplift_train"]["client_id"]) if "client_id" in dfs["uplift_train"].columns else set()
        clients_ids = set(dfs["clients"]["client_id"])
        overlap = len(train_ids & clients_ids)
        report_sections.append(
            f"- `uplift_train.client_id` ∩ `clients.client_id`: {overlap:,} "
            f"(uplift_train totali: {len(train_ids):,})"
        )

    if "uplift_test" in dfs and "clients" in dfs:
        test_ids = set(dfs["uplift_test"]["client_id"]) if "client_id" in dfs["uplift_test"].columns else set()
        clients_ids = set(dfs["clients"]["client_id"])
        overlap = len(test_ids & clients_ids)
        report_sections.append(
            f"- `uplift_test.client_id` ∩ `clients.client_id`: {overlap:,} "
            f"(uplift_test totali: {len(test_ids):,})"
        )

    if "uplift_train" in dfs and "uplift_test" in dfs:
        train_ids = set(dfs["uplift_train"]["client_id"]) if "client_id" in dfs["uplift_train"].columns else set()
        test_ids = set(dfs["uplift_test"]["client_id"]) if "client_id" in dfs["uplift_test"].columns else set()
        overlap = len(train_ids & test_ids)
        report_sections.append(
            f"- `uplift_train.client_id` ∩ `uplift_test.client_id`: {overlap:,} "
            f"(atteso: 0, devono essere disgiunti)"
        )

    report_sections.append("")

    REPORT_PATH.write_text("\n".join(report_sections), encoding="utf-8")
    print(f"\nReport salvato in: {REPORT_PATH}")
    print("Audit completato. Usa data_audit.md come riferimento per preprocessing e modeling.")


if __name__ == "__main__":
    main()
