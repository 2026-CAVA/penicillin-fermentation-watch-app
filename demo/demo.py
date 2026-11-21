"""Presentation walk-through of the API. Start the server first (python -m main serve), then:
python demo/demo.py [--url http://127.0.0.1:8000] [--no-pause]
Each step prints one API call and a short summary; Enter moves to the next step.
run.csv = fault batch 92's input columns with the batch number removed (for `python -m main predict`).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--no-pause", action="store_true")
a = p.parse_args()


def get(path):
    print(f"\n  GET {path}")
    with urlopen(a.url + path) as r:
        return json.load(r)


def step(n, title):
    if n > 1 and not a.no_pause:
        input("\n[Enter] next")
    print(f"\n{'=' * 70}\n{n}. {title}\n{'=' * 70}")


def batch_line(b):
    lo, hi = b["predictedRange"] or (None, None)
    rng = f" ({lo:.0f}-{hi:.0f})" if lo is not None else ""
    pred = f"{b['predictedFinal']:.0f} kg{rng}" if b["predictedFinal"] is not None else "not yet"
    return (f"  batch {b['id']:>3}  t={b['elapsedH']:>5.1f} h  state={b['state']:<8} R={b['latestScore']:>3.0f}  "
            f"yield pred {pred}" + (f"  measured {b['finalMeasured']:.0f} kg" if b.get("finalMeasured") else ""))


step(1, "Model and cross-validated performance")
m = get("/model")
print(f"  {m['title']}\n  models: {m['model']}  features {m['n_features']}, normal batches {m['n_train']}")
for k, v in m["cv"]["alarms"].items():
    print(f"  alarm {k:<9} normal batches alarmed {v['normal_alarmed_share']['mean']:.0%}, "
          f"faults detected {v['fault_detected_share']['mean']:.0%}, {v['delay_h_median']['mean']:.1f} h after onset")
for h, c in m["cutoffs"].items():
    print(f"  yield at {h:>3} h: {c['model']:<10} CV R2 {c['cv_r2']:.2f}  80% range +-{c['interval_halfwidth']:.0f} kg")

step(2, "Batch list (demo batches 6 normal, 92 fault, 62 normal APC)")
bs = {b["id"]: b for b in get("/batches")}
for i in (6, 92, 62):
    print(batch_line(bs[i]))
print(f"  ... {sum(b['state'] != 'normal' for b in bs.values())} of {len(bs)} batches ended with an alarm")

step(3, "Replay normal batch 6 over time (only data up to t is used)")
for t in (46, 115, 161):
    print(batch_line(get(f"/batches/6?t={t}")))

step(4, "Fault batch 92: when does the alarm fire?")
for e in get("/batches/92/events")[:5]:
    print(f"  {e['start']:>5.1f}-{e['end']:<5.1f} h  {e['grade']:<8} {e['message']}")
d = get("/batches/92?t=100")
print(batch_line(d) + f"\n  {d['explanation']}")

step(5, "Why this yield prediction (contributions) and penicillin forecast")
for s in get("/batches/92/shap?t=115"):
    print(f"  {s['value']:+7.0f} kg  {s['label']}")
f = get("/batches/92/forecast?t=100")
print(f"  last measured {f['last_measured']}")
for q in f["path"][::5]:
    print(f"  P(t+{q['h']:g} h) = {q['p']:.2f} g/L  (80% {q['lo']:.2f}-{q['hi']:.2f})")

step(6, "A new run that is not in the data (CSV in, prediction out)")
cmd = [sys.executable, "-m", "main", "predict", str(Path(__file__).with_name("run.csv")), "--t", "100"]
print("  " + " ".join(cmd[1:]), flush=True)
subprocess.run(cmd, cwd=Path(__file__).parents[1])

step(7, "Normal 6 vs fault 92")
for b in get("/compare?ids=6,92")["batches"]:
    print(batch_line(b) + f"  group={b['group']}")
print("\ndone")
