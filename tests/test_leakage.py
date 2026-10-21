"""Regression test for the v0 'zero leakage' artefact (see docs/leakage_analysis.md).

v0 measured the other camera's coverage only once, at onset. A camera-anchored wrist occluder
re-aligns to the wrist->cube ray every step, so as the hand moves over the cube the plate
ends up above the cube and hides it from the third-person camera too. Coverage is now
measured in both cameras on every step the occluder is in the scene.
"""

import numpy as np

from occlubench.conditions import Condition
from occlubench.env import OccluEnv
from occlubench.policies import ScriptedOracle


def test_wrist_occluder_leak_is_measured_during_approach():
    n = 4
    env = OccluEnv(n_envs=n, horizon=60, render="none")
    obs = env.reset(list(range(n)), [Condition("wrist", 1.0, 0.0, 0.5)] * n)
    pol = ScriptedOracle()
    pol.reset(n)
    done = False
    while not done:
        obs, done, _ = env.step(pol.act(obs, env))
    st = env.coverage_stats()
    assert np.all(st["coverage_onset"] > 0.95)
    # Third-person leakage starts low but becomes large once the hand is above the cube.
    assert np.all(st["coverage_other_max"] > 0.5)
    env.close()
