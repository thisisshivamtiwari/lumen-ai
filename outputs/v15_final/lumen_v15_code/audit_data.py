"""Read-only audit of the released competition tables; outputs stay local."""
from pathlib import Path
import argparse
import json
import pandas as pd
import pyarrow.parquet as pq


def audit(directory):
    directory = Path(directory)
    out = {"directory": str(directory), "tables": {}}
    for name in ("transactions", "transfers", "blocks"):
        p = directory / f"{name}.parquet"
        meta = pq.ParquetFile(p)
        out["tables"][name] = {
            "rows": meta.metadata.num_rows,
            "schema": str(meta.schema_arrow),
        }
    tx = pq.read_table(directory / "transactions.parquet").to_pandas()
    tr = pq.read_table(directory / "transfers.parquet").to_pandas()
    rates = pd.read_csv(directory / "market_rates.csv")
    out["rate_columns"] = rates.columns.tolist()
    out["rate_rows"] = len(rates)
    out["transaction_duplicates"] = int(tx.TRANSACTION_HASH.duplicated().sum())
    out["transfer_key_duplicates"] = int(tr.duplicated(["TRANSACTION_HASH", "TRANSFER_INDEX"]).sum())
    out["orphan_transfer_rows"] = int((~tr.TRANSACTION_HASH.isin(tx.TRANSACTION_HASH)).sum())
    counts = tr.groupby("TRANSACTION_HASH").size().reindex(tx.TRANSACTION_HASH, fill_value=0)
    out["legs_per_transaction"] = {str(k): int(v) for k, v in counts.value_counts().sort_index().items()}
    out["timestamp_range"] = [int(tx.TIMESTAMP.min()), int(tx.TIMESTAMP.max())]
    out["timestamp_utc"] = [str(pd.to_datetime(t, unit="s", utc=True)) for t in out["timestamp_range"]]
    out["transaction_types"] = tx.TRANSACTION_TYPE.value_counts(dropna=False).to_dict() if "TRANSACTION_TYPE" in tx else {}
    out["null_counts"] = {"transactions": tx.isna().sum().to_dict(), "transfers": tr.isna().sum().to_dict()}
    out["asset_counts"] = tr.groupby(["TOKEN_ADDRESS", "SYMBOL"]).size().rename("rows").reset_index().to_dict("records")
    out["rate_key_duplicates"] = int(rates.duplicated(["TIMESTAMP", "ASSET"]).sum())
    label_file = directory / "labels.csv"
    if label_file.exists():
        labels = pd.read_csv(label_file)
        out["labels"] = {
            "columns": labels.columns.tolist(),
            "rows": len(labels),
            "duplicates": int(labels.TRANSACTION_HASH.duplicated().sum()),
            "missing_transaction_ids": int((~tx.TRANSACTION_HASH.isin(labels.TRANSACTION_HASH)).sum()),
            "extra_transaction_ids": int((~labels.TRANSACTION_HASH.isin(tx.TRANSACTION_HASH)).sum()),
            "positive_count": int(labels.is_suspicious.sum()),
            "prevalence": float(labels.is_suspicious.mean()),
            "family_difficulty": labels.groupby(["family", "difficulty", "is_suspicious"]).size().rename("count").reset_index().to_dict("records"),
            "scenario_count": int(labels.scenario_id.nunique()),
            "cohort_count": int(labels.cohort_id.nunique()),
        }
    for key in ("transaction_duplicates", "transfer_key_duplicates", "orphan_transfer_rows", "rate_key_duplicates"):
        assert out[key] == 0, (directory, key, out[key])
    assert counts.between(1, 3).all()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/data_audit.json"))
    args = parser.parse_args()
    result = {name: audit(path) for name, path in {
        "train": "data/training/train",
        "practice": "data/training/validation",
        "test": "data/test",
    }.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=int) + "\n")
    for name, summary in result.items():
        print(name, {k: v for k, v in summary.items() if k not in {"tables", "null_counts", "asset_counts", "labels"}})
        for table, info in summary["tables"].items():
            print(table, info["rows"], info["schema"])
        if "labels" in summary:
            print("labels", json.dumps(summary["labels"], indent=2))
