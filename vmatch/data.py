"""Load and clean the NYC volunteer opportunities data."""
from pathlib import Path
import re
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "volunteer_opportunities.csv"


def clean_text(s: str) -> str:
    s = re.sub(r"https?://\S+|www\.\S+", " ", str(s))       # drop URLs
    s = re.sub(r"\S+@\S+", " ", s)                           # drop emails
    return re.sub(r"\s+", " ", s).strip()


def load(path: Path = DATA) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.status == "approved"].copy()
    for c in ["title", "summary", "org_title"]:
        d[c] = d[c].fillna("").map(clean_text)
    # same listing posted more than once: keep the first
    d = d.drop_duplicates(subset=["title", "summary"]).reset_index(drop=True)
    d["category"] = d["category_desc"].fillna("Uncategorised")
    d["text"] = d["title"] + ". " + d["summary"]
    d["doc_id"] = d["opportunity_id"].astype(str)
    assert d.doc_id.is_unique, "opportunity ids must be unique"
    assert (d.text.str.len() > 10).all(), "empty listing text"
    return d[["doc_id", "title", "summary", "org_title", "category", "recurrence_type", "text", "hits"]]
