"""Rank all trained algorithms by test macro-F1 (accuracy is misleading: ~85% of wafers are 'none')."""
import json
from pathlib import Path

rows = [json.loads(p.read_text()) for p in sorted(Path(__file__).parent.joinpath("results").glob("*.json")) if p.stem != "summary" and not p.stem.startswith("ensemble")]
rows.sort(key=lambda r: r["test"]["macro_f1"], reverse=True)
lines = ["| Rank | Model | Macro-F1 | Balanced acc | Accuracy | Params | Train time (s) | Best epoch |", "|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    t = r["test"]
    lines.append(f"| {i} | {r['name']} | {t['macro_f1']:.2%} | {t['balanced_accuracy']:.2%} | {t['accuracy']:.2%} | {r['params']:,} | {r['train_seconds']:.0f} | {r['best_epoch']} |")
classes = list(rows[0]["per_class_f1"])
lines += ["", "Per-class F1:", "", "| Model | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
for r in rows:
    lines.append(f"| {r['name']} | " + " | ".join(f"{r['per_class_f1'][c]:.2f}" for c in classes) + " |")
lines += ["", "*Train time is wall-clock. googlenet_v2_64 includes laptop suspend time (and used batch size 64), so its value is not a compute-time measurement.*"]
text = "\n".join(lines)
Path(__file__).parent.joinpath("results", "summary.md").write_text(text + "\n")
print("\n" + text + f"\n\nBest: {rows[0]['name']}")
