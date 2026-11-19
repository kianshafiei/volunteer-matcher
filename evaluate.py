"""Evaluate four ranking methods on 30 volunteer profiles.

Relevance labels (data/qrels.json) are graded 0/1/2 on the pooled top-10 of every method,
so every result shown at depth 10 has a judgment. See README for how labels were made.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from vmatch.data import load
from vmatch.matcher import Matcher

ROOT = Path(__file__).parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
K = 10
METHODS = {"bm25": "Keyword (BM25)", "dense": "Semantic (MiniLM)", "hybrid": "Hybrid (RRF)",
           "full": "Hybrid + preference + diversity"}


def ndcg(gains, ideal, k=K):
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (np.asarray(gains[:k]) * disc[:len(gains[:k])]).sum()
    ig = np.sort(ideal)[::-1][:k]
    idcg = (ig * disc[:len(ig)]).sum()
    return dcg / idcg if idcg > 0 else 0.0


def main():
    df = load()
    m = Matcher(df)
    profiles = json.load(open(ROOT / "data/profiles.json"))
    qrels = json.load(open(ROOT / "data/qrels.json"))
    cat = dict(zip(df.doc_id, df.category)); org = dict(zip(df.doc_id, df.org_title))
    rec = dict(zip(df.doc_id, df.recurrence_type))
    rows, shown = [], {k: set() for k in METHODS}
    for p in profiles:
        q, rel = p["text"], qrels[p["id"]]
        for meth in METHODS:
            ids = [x.doc_id for x in m.search(q, K, meth)]
            missing = [i for i in ids if i not in rel]
            assert not missing, f"unjudged result {missing} for {p['id']} {meth}"
            g = [rel[i] for i in ids]
            shown[meth].update(ids)
            first = next((r for r, x in enumerate(g) if x > 0), None)
            rows.append({
                "profile": p["id"], "method": meth,
                "P@5": np.mean([x > 0 for x in g[:5]]),
                "nDCG@10": ndcg(g, list(rel.values())),
                "MRR": 0 if first is None else 1 / (first + 1),
                "category P@5": np.mean([cat[i] == p["category"] for i in ids[:5]]) if p["category"] else np.nan,
                "distinct orgs @10": len({org[i] for i in ids}),
                "one-off share @10": np.mean([rec[i] == "onetime" for i in ids]),
            })
    r = pd.DataFrame(rows)
    summary = r.groupby("method").mean(numeric_only=True).loc[list(METHODS)]
    summary["catalogue coverage"] = [len(shown[k]) / len(df) for k in METHODS]
    summary.index = [METHODS[k] for k in summary.index]

    # paired bootstrap: full vs keyword and full vs semantic on nDCG@10
    rng = np.random.default_rng(0)
    piv = r.pivot(index="profile", columns="method", values="nDCG@10")
    boot = {}
    for new, base in [("hybrid", "bm25"), ("hybrid", "dense"), ("full", "bm25"), ("full", "hybrid")]:
        d = (piv[new] - piv[base]).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        boot[f"{new} minus {base}"] = {"mean": round(float(d.mean()), 3),
                                      "95% CI": [round(float(np.percentile(bs, 2.5)), 3),
                                                 round(float(np.percentile(bs, 97.5)), 3)],
                                      "profiles improved / worse": [int((d > 0).sum()), int((d < 0).sum())]}
    oneoff = r[r.profile == "q26"].set_index("method")["one-off share @10"].round(2).to_dict()
    res = {"n_profiles": len(profiles), "n_opportunities": len(df), "n_judgments": sum(len(v) for v in qrels.values()),
           "summary": summary.round(3).to_dict(orient="index"), "bootstrap_nDCG": boot,
           "one-off request (q26), share of one-off results": oneoff}
    json.dump(res, open(OUT / "results.json", "w"), indent=2)
    r.to_csv(OUT / "per_profile.csv", index=False)
    print(summary.round(3).to_string()); print(json.dumps(boot, indent=1)); print(oneoff)


if __name__ == "__main__":
    main()
