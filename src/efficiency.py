"""Model-side efficiency: parameters, FLOPs, latency (batch 1, GPU/CPU), throughput, peak memory.
python efficiency.py --models C4-WR-TGA FT-Transformer ... """
import os, sys, json, time, argparse
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
from mldata import Pool, W
import train_dl, bench, tab, nets

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results2")


def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()


def bench_model(model, x, dev, n=100, warm=20):
    model.eval()
    with torch.no_grad():
        for _ in range(warm): model(x)
        sync(dev); t0 = time.perf_counter()
        for _ in range(n): model(x)
        sync(dev)
    return 1e3 * (time.perf_counter() - t0) / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--cases", nargs="+", default=["case14", "case39", "case118"])
    ap.add_argument("--out", default="efficiency_models.jsonl")
    a = ap.parse_args()
    out = os.path.join(RES, a.out)
    for case in a.cases:
        tr = Pool(case, "train_a")
        for name in a.models:
            if name in bench.RULES or name in bench.TABULAR or name == "C5-WR-Boost":
                continue
            base, cfg = (bench.CANDIDATES[name] if name in bench.CANDIDATES else (name, {}))
            cfg = dict(cfg)
            tj = os.path.join(RES, "tuned", f"{name}.json")
            if os.path.exists(tj): cfg.update(json.load(open(tj))["best_params"])
            row = dict(name=name, case=case, m=tr.m)
            for dname in ("cuda", "cpu"):
                dev = torch.device(dname)
                if dname == "cuda" and not torch.cuda.is_available(): continue
                model = train_dl.make_model(base, tr, cfg).to(dev)
                x1 = torch.randn(1, W, tr.m, 5, device=dev).abs(); x256 = torch.randn(256, W, tr.m, 5, device=dev).abs()
                row["params"] = sum(p.numel() for p in model.parameters())
                if dname == "cuda": torch.cuda.reset_peak_memory_stats()
                row[f"lat1_{dname}_ms"] = bench_model(model, x1, dev, n=100 if dname == "cuda" else 30)
                nb = 20 if dname == "cuda" else 5
                tb = bench_model(model, x256, dev, n=nb, warm=3)
                row[f"thr_{dname}_samples_s"] = 256 / (tb / 1e3)
                if dname == "cuda": row["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1e6
                if dname == "cpu":
                    try:
                        with FlopCounterMode(display=False) as fc:
                            model.eval()
                            with torch.no_grad(): model(x1)
                        row["flops_per_sample"] = fc.get_total_flops()
                    except Exception as e:
                        row["flops_per_sample"] = None
            with open(out, "a") as f: f.write(json.dumps(row) + "\n")
            print(name, case, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
