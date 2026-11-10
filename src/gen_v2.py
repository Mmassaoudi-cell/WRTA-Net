"""Generate the v2 pools: augmented train/val, shifted validation + test pools, and two independent data replicates."""
import subprocess, sys, os
cases = sys.argv[1:] or ["case14", "case39", "case118"]
jobs = []
for c in cases:
    k = 0
    def add(pool, n, tag, off, *args):
        jobs.append((c, pool, n, tag, off, list(args)))
    add("train", 600, "train_a", 500_000, "--aug")
    add("val", 200, "val_a", 500_000, "--aug")
    # shifted validation pools (model selection) and shifted test pools (final evaluation)
    for role, pool, base in [("val", "val", 600_000), ("test", "rob", 1_000_000)]:
        off = base
        for tag, args in [("pmu10", ["--pmuscale", "10"]), ("adapt", ["--stealth", "2"]), ("off", ["--offprior", "1.0"]), ("topo", ["--outage"]), ("topof", ["--outage", "--topoknown"])]:
            add(pool, 200, f"{role}_{tag}" if role == "val" else f"rob_{tag}", off, *args); off += 10_000
        if role == "test":
            for tag, args in [("pmu3", ["--pmuscale", "3"]), ("pmu30", ["--pmuscale", "30"]), ("pmu100", ["--pmuscale", "100"])]:
                add(pool, 200, f"rob_{tag}", off, *args); off += 10_000
    for rep in (1, 2):
        off = 2_000_000 * rep
        add("train", 600, f"r{rep}_train_a", off, "--aug"); add("val", 200, f"r{rep}_val_a", off, "--aug"); add("test", 400, f"r{rep}_test", off)
for c, pool, n, tag, off, args in jobs:
    if os.path.exists(os.path.join("..", "data", f"{c}_{tag}.npz")):
        continue
    env = dict(os.environ); env["GEN_SEEDOFF"] = str(off)
    subprocess.run([sys.executable, "gen_data.py", c, pool, str(n), "--tag", tag] + args, check=True, env=env)
