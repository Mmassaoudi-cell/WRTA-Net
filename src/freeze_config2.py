import json, os, hashlib, yaml, datetime, glob
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import bench
tuned = json.load(open(os.path.join(ROOT, "results2", "tuned", "V2c.json")))
base, cfg = bench.CANDIDATES["V2c"]
cfg = dict(cfg); cfg.update(tuned["best_params"])
def h(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
out = {
 "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "note": "Frozen before the final evaluation of the proposed model on any test pool. Selection used validation pools only (augmented validation pool + five shifted validation pools).",
 "proposed_model": {
   "name": "WRTA-Net", "internal_id": "V2c",
   "architecture": {"front_end": "PMU-only linear LSE residual (non-iterative), whitened by nominal sigma; clone feature; log robust noise scale; topology-processor flag; Stage-1 PTDF bit",
                    "encoder": "per-meter shared diagonal SSM (S4D) conv, FiLM on log robust scale", "temporal_layers": cfg["layers"],
                    "spatial_mixing": "physics-biased multi-head attention (heads=4, beta init 2.0), layers=%d; no local graph mixing" % cfg["glayers"],
                    "vulnerability_prior_dropout": cfg.get("prior_drop", 0.0), "hidden_dim_d": cfg["d"], "heads": ["meter logit", "4-class logit"]},
   "hyperparameters": {k: v for k, v in cfg.items()},
   "optimizer": "AdamW", "scheduler": "OneCycleLR(pct_start=0.15)", "grad_clip": 2.0,
   "loss": "CE(class, weights=(N/(4 n_c))^0.5) + lam * BCE(meters, pos_weight=clip(sqrt(neg/pos),1,10)); lam=%s" % cfg["lam"],
   "epochs": 20, "early_stopping": "none; checkpoint of best validation epoch", "augmentation": "robust-training mixture in the training pool (PMU noise x1-x30, topology errors, three replay recipes, 40% off-prior attackers)",
   "meter_threshold": "tau = argmax validation attacked-meter F1 (validation only)",
 },
 "data": {"train_pool": "train_a (600 scenarios/system, augmented mixture)", "validation_pool": "val_a (200)", "test_pools": "test (400, nominal) + 10 shifted pools (200 each) + 2 independent replicates",
          "files_sha256_16": {os.path.basename(p): h(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "*.npz")))}},
 "seeds_main": [0, 1, 2, 3, 4], "code_sha256_16": {os.path.basename(p): h(p) for p in sorted(glob.glob(os.path.join(ROOT, "src", "*.py"))) if "backup" not in p},
 "tuning": {"budget": "8 Optuna TPE trials on IEEE-14 + NE-39 validation (mean score), 10 epochs; every learned benchmark: 3-6 trials under the same objective", "tuned_files": "results2/tuned/*.json"},
}
yaml.safe_dump(out, open(os.path.join(ROOT, "FINAL_MODEL_CONFIG.yaml"), "w"), sort_keys=False, width=140)
print("frozen", cfg)
