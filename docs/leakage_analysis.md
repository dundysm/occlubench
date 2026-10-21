# Why a wrist occluder hurt a third-person-only policy (v0) — root cause

**v0 observation** ([results_v0_cpu.md](results_v0_cpu.md)): with `Condition("wrist", 1.0, onset=0.0)`, the
memoryless `ColorCuePolicy`, which only looks at the third-person camera, lost about 44 points of success, yet v0 reported
third-person leakage `coverage_other = 0.00`.

## Root cause

Two things combined:

1. **The occluder is camera-anchored.** Every step it re-aligns to the ray from the wrist camera to the cube, at 45% of
   that distance.
2. **v0 measured leakage only once, at onset.** At t=0 the hand is high above the start position, so the wrist→cube ray
   runs at an angle and the plate is not on the third-person line of sight. As the policy approaches the cube, the hand
   ends up above it, and the wrist→cube ray becomes nearly vertical. The plate then sits a few centimetres above the cube
   and hides it from the third-person camera too. The policy re-localises the cube from the third-person view while
   approaching, so it loses sight of the cube, holds still, and waits until the occluder leaves.

A per-step trace from v0 (4 failing seeds, third-person coverage measured every 5 steps by CPU segmentation):

```
t=0..5   tp_leak 0.00 0.00 0.00 0.00
t=10     tp_leak 0.00 0.00 0.45 0.80
t=20     tp_leak 0.91 0.68 1.00 1.00
t=25..45 tp_leak ~1.00 for all
t=50     tp_leak 0.00 (occluder leaves)
```

## Fix in v0.2

* Coverage is now measured in **both** cameras on **every** step the occluder is in the scene, using the vectorised
  geometric estimator. Records carry `coverage` and `coverage_other` (means over the active window),
  `coverage_other_max` (worst step) and `coverage_onset` (the calibrated value).
* New `Condition(anchor="world")` freezes the occluder's pose in the world at onset, like an object that moves in and
  stops. `anchor="camera"` (the default) keeps target-camera severity constant as the wrist camera moves, but leaks into
  the other camera. Both modes are now reported honestly.
* Regression test: `tests/test_leakage.py`.

## Re-run (v0.2, gantry, ColorCue, 16 seeds, 48×48, horizon 160, CPU)

| condition | coverage (mean target / mean leak into other cam) | clean SR | pert SR | paired drop [95% CI] |
|---|---|---|---|---|
| wrist, sev 1.0, onset 0, anchor=camera | 0.77 / **0.64** | 0.94 | 0.31 | +0.625 [+0.375, +0.875] |
| wrist, sev 1.0, onset 0, anchor=world | 0.15 / 0.04 | 0.94 | 1.00 | −0.062 [−0.188, +0.000] |
| third-person, sev 1.0, onset 0 | 0.77 / 0.00 | 0.94 | 0.75 | +0.188 [+0.000, +0.375] |

The camera-anchored wrist occluder leaks about 64% into the third-person view on average, and that leak fully explains
the drop. When the occluder is world-anchored, leakage falls to about 4% and the third-person-only policy is unaffected,
as expected. (Clean success here is 15/16 because the third-person field of view widened from 50° to 58° in v0.2 to fit
the Panda.)

**Takeaway for the benchmark:** a physical occluder is not camera-isolated. Use `coverage_other` / `coverage_other_max`
to separate "the targeted camera was blocked" from "both cameras were blocked". For wrist-vs-third-person comparisons,
also report `anchor="world"`.
