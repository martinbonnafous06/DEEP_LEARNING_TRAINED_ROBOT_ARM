"""scripts/test_libero.py — Verification script for LIBERO environment and MuJoCo simulation.

Runs a fast diagnostic to verify:
1. LIBERO package and benchmark suite registry.
2. MuJoCo offscreen rendering and BDDL domain parsing.
3. Environment reset and control step execution.

Usage:
    python scripts/test_libero.py
"""

from __future__ import annotations

import os
import sys
import time

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] GPU Device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[WARNING] PyTorch check failed: {e}")

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    print("[OK] LIBERO package loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to import LIBERO: {e}")
    sys.exit(1)

def run_test():
    print("\n--- Testing LIBERO Suite Registry ---")
    b_dict = benchmark.get_benchmark_dict()
    print(f"[OK] Available suites ({len(b_dict)}): {list(b_dict.keys())}")

    print("\n--- Testing Task Loading & BDDL ---")
    suite_name = "libero_10"
    task_suite = b_dict[suite_name]()
    task_id = 0
    task = task_suite.get_task(task_id)
    print(f"[OK] Suite: {suite_name} (Total tasks: {task_suite.get_num_tasks()})")
    print(f"[OK] Task {task_id}: {task.name}")
    print(f"[OK] Instruction: \"{task.language}\"")

    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[OK] BDDL domain file: {bddl_file}")

    print("\n--- Testing MuJoCo OffScreen Simulation ---")
    start_t = time.time()
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    obs = env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    obs_keys = list(obs.keys())
    print(f"[OK] Environment reset succeeded in {time.time() - start_t:.2f}s")
    print(f"[OK] Observation keys captured: {obs_keys[:5]}")

    dummy_action = [0.0] * 7
    obs, reward, done, info = env.step(dummy_action)
    print(f"[OK] Simulation step executed! Reward: {reward}, Done: {done}")
    env.close()

    print("\n==================================================")
    print(">>> SUCCESS: LIBERO RUNS 100% SMOOTHLY! <<<")
    print("==================================================")

if __name__ == "__main__":
    run_test()
