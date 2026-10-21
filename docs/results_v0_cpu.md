# v0 example results (CPU, illustrative only)

`MUJOCO_GL=osmesa python scripts/run_grid.py --policy colorcue --seeds 16 --n-envs 16 --image 48 48 --onsets 0.0 0.15 --severities 0.5 1.0`, horizon 160. CPU MuJoCo with the CPU renderer on a CPU-only box, about 17 minutes. The n=16 seeds give wide CIs; the protocol target is at least 1,000 pairs per cell on GPU.

| condition | n | coverage (other cam) | clean SR | pert SR | paired drop [95% CI] | P(pert ok given clean ok) [95% CI] | recovery steps [95% CI] |
|---|---|---|---|---|---|---|---|
| third_person_sev0.50_on0.00_dur0.25 | 16 | 0.54 (0.00) | 1.00 | 0.75 | +0.250 [+0.062, +0.500] | +0.750 [+0.500, +0.938] | +13.083 [+3.167, +23.667] |
| third_person_sev0.50_on0.15_dur0.25 | 16 | 0.62 (0.00) | 1.00 | 1.00 | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] |
| third_person_sev1.00_on0.00_dur0.25 | 16 | 1.00 (0.00) | 1.00 | 0.69 | +0.312 [+0.125, +0.562] | +0.688 [+0.438, +0.875] | +46.455 [+45.455, +47.455] |
| third_person_sev1.00_on0.15_dur0.25 | 16 | 1.00 (0.00) | 1.00 | 1.00 | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] |
| wrist_sev0.50_on0.00_dur0.25 | 16 | 0.48 (0.00) | 1.00 | 0.69 | +0.312 [+0.125, +0.562] | +0.688 [+0.438, +0.875] | +30.091 [+19.452, +39.364] |
| wrist_sev0.50_on0.15_dur0.25 | 16 | 0.32 (0.97) | 1.00 | 1.00 | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] |
| wrist_sev1.00_on0.00_dur0.25 | 16 | 1.00 (0.00) | 1.00 | 0.56 | +0.438 [+0.188, +0.688] | +0.562 [+0.312, +0.812] | +44.000 [+41.889, +46.111] |
| wrist_sev1.00_on0.15_dur0.25 | 16 | 0.58 (0.98) | 1.00 | 1.00 | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] |

How to read this:
- **Onset 0.00:** the occluder is present while `ColorCuePolicy` (memoryless) is still looking for the cube, so it waits and sometimes runs out of horizon. The drop grows with severity (third-person: +0.25 at 50%, +0.31 at 100%).
- **Onset 0.15:** by then the policy has already localised and grasped the cube, so occlusion has no effect. This is the timing axis doing its job, since the same occluder is harmless or harmful depending on onset.
- **Wrist occluder leaks into the third-person view.** At onset 0.15 (`coverage_other` 0.97–0.98), the wrist-targeted occluder also hides the cube from the third-person camera. A physical occluder is not camera-isolated, and the benchmark reports this rather than hiding it. At onset 0.00, the wrist occluder still hurts this third-person-only policy even though measured leakage at calibration was 0.00. Separating that effect is on the v0.1 list.
- Achieved coverage differs from the target because coverage is measured on 64×64 segmentation renders and the cube may already be in the gripper.
