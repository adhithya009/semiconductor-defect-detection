"""Ensemble saved V2 probabilities and tune per-class decision weights on validation only; report test once."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "_v2"
probs = {p.stem: np.load(p) for p in sorted((ROOT / "results" / "probs").glob(f"*{tag}.npz"))}
classes = list(json.loads(next(iter((ROOT / "results").glob(f"*{tag}.json"))).read_text())["per_class_f1"])
yv, yt = next(iter(probs.values()))["val_y"], next(iter(probs.values()))["test_y"]


def score(p, y, w=None):
    pred = (p * (1 if w is None else w)).argmax(1)
    return f1_score(y, pred, average="macro", zero_division=0), pred


def tune(pv):
    """Coordinate ascent over per-class multiplicative weights, maximising validation macro-F1."""
    w, best = np.ones(pv.shape[1]), score(pv, yv)[0]
    for _ in range(2):
        for c in range(len(w)):
            for g in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0):
                trial = w.copy(); trial[c] = w[c] * g
                s = score(pv, yv, trial)[0]
                if s > best + 1e-5: best, w = s, trial
    return w


rows = []
combos = [(n,) for n in probs] + [c for r in range(2, len(probs) + 1) for c in itertools.combinations(probs, r)]
for combo in combos:
    pv = sum(probs[n]["val_probs"] for n in combo) / len(combo)
    pt = sum(probs[n]["test_probs"] for n in combo) / len(combo)
    w = tune(pv)
    for label, weights in (("avg", None), ("avg+thr", w)):
        f1, pred = score(pt, yt, weights)
        rows.append({"models": "+".join(n.replace(tag, "") for n in combo), "mode": label, "val_f1": score(pv, yv, weights)[0],
                     "macro_f1": f1, "balanced_acc": balanced_accuracy_score(yt, pred), "accuracy": accuracy_score(yt, pred),
                     "per_class_f1": dict(zip(classes, f1_score(yt, pred, average=None, zero_division=0).round(3).tolist()))})
rows.sort(key=lambda r: r["val_f1"], reverse=True)  # select by validation, never by test
(ROOT / "results" / f"ensemble{tag}.json").write_text(json.dumps(rows, indent=2))
print("| models | mode | val F1 | test macro-F1 | bal acc | acc |\n|---|---|---|---|---|---|")
for r in rows[:8]:
    print(f"| {r['models']} | {r['mode']} | {r['val_f1']:.2%} | {r['macro_f1']:.2%} | {r['balanced_acc']:.2%} | {r['accuracy']:.2%} |")
print("\nSelected (best val):", rows[0]["models"], rows[0]["mode"], rows[0]["per_class_f1"])
