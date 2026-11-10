import json, sys, pandas as pd, numpy as np
fn = sys.argv[1] if len(sys.argv) > 1 else "../results2/sel_v2.jsonl"
rows = []
for l in open(fn):
    r = json.loads(l)
    d = dict(name=r["name"], case=r["case"], seed=r["seed"], A=r["val_macro_f1"], B=r["val_b_f1_att"], params=r["params"])
    for t, v in r.get("extra_val", {}).items():
        d[f"A_{t}"] = v["A_macroF1"]; d[f"B_{t}"] = v["B_F1att"]
    rows.append(d)
df = pd.DataFrame(rows)
cols = [c for c in df.columns if c not in ("name", "case", "seed")]
pd.set_option("display.width", 250)
g = df.groupby(["name", "case"])[cols].mean().round(3)
print(g.to_string())
sc = df.assign(S=df[[c for c in cols if c.startswith(("A", "B")) and c != "params"]].mean(1)).groupby("name").S.mean().sort_values(ascending=False)
print(sc.round(4))
