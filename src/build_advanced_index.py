"""Build the multimodal index (text, structured tables, diagrams, charts) and benchmark for RL.pdf (108 pages)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from pypdf import PdfReader

DAY7_ROOT = Path(__file__).resolve().parent.parent
if str(DAY7_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(DAY7_ROOT / "starter"))

from baseline import validate  # noqa: E402

SOURCE_ID = "rl-compendium-108p"
SOURCE_FILE = "RL.pdf"


def _paper_for_page(page: int) -> str:
    if 1 <= page <= 19:
        return "Paper 1: AWR (Peng et al., 2019)"
    if 20 <= page <= 61:
        return "Paper 2: Harness VLA (Zhang et al., 2026)"
    if 62 <= page <= 81:
        return "Paper 3: AMP (Peng et al., 2021)"
    if 82 <= page <= 88:
        return "Paper 4: Forward-Backward RL (Edwards et al., 2018)"
    return "Duplicate Attachment: AMP (Peng et al., 2021)"


def _structured_tables() -> list[dict[str, Any]]:
    """Curated, original-PDF-verified structured tables across all papers in RL.pdf."""
    return [
        {
            "id": f"{SOURCE_ID}::p008::table::001",
            "page": 8,
            "table_id": "AWR Table 1",
            "bbox": [48, 420, 564, 595],
            "units": "Average episode return ± 1 SD across 5 random seeds",
            "text": (
                "[AWR Table 1 | Page 8] Table 1: Final returns for different algorithms on the OpenAI Gym tasks "
                "(± 1 standard deviation across 5 random seeds).\n"
                "| Task | TRPO | PPO | DDPG | TD3 | SAC | RWR | AWR (Ours) |\n"
                "|---|---|---|---|---|---|---|---|\n"
                "| Ant-v2 | 2901 ± 85 | 1161 ± 389 | 72 ± 1550 | 4285 ± 671 | 5909 ± 371 | 181 ± 19 | 5067 ± 256 |\n"
                "| HalfCheetah-v2 | 3302 ± 428 | 4920 ± 429 | 10563 ± 382 | 4309 ± 1238 | 9297 ± 1206 | 1400 ± 370 | 9136 ± 184 |\n"
                "| Hopper-v2 | 1880 ± 337 | 1391 ± 304 | 855 ± 282 | 935 ± 489 | 2769 ± 552 | 605 ± 114 | 3405 ± 121 |\n"
                "| Humanoid-v2 | 552 ± 9 | 695 ± 59 | 4382 ± 423 | 81 ± 17 | 8048 ± 700 | 509 ± 18 | 4996 ± 697 |\n"
                "| LunarLander-v2 | 104 ± 94 | 121 ± 49 | - | - | - | 185 ± 23 | 229 ± 2 |\n"
                "| Walker2d-v2 | 2765 ± 168 | 2617 ± 362 | 401 ± 470 | 4212 ± 427 | 5805 ± 587 | 406 ± 64 | 5813 ± 483 |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p010::table::001",
            "page": 10,
            "table_id": "AWR Table 2",
            "bbox": [48, 610, 305, 745],
            "units": "Normalized return in [0, 1] ± SD",
            "text": (
                "[AWR Table 2 | Page 10] Table 2: Performance statistics of algorithms on the motion imitation tasks "
                "(34 DoF humanoid and 82 DoF dog). Returns are normalized between minimum and maximum possible returns per episode.\n"
                "| Task | PPO | RWR | AWR (Ours) |\n"
                "|---|---|---|---|\n"
                "| Humanoid: Cartwheel | 0.76 ± 0.02 | 0.03 ± 0.01 | 0.78 ± 0.07 |\n"
                "| Humanoid: Spinkick | 0.70 ± 0.02 | 0.05 ± 0.03 | 0.77 ± 0.04 |\n"
                "| Dog: Canter | 0.76 ± 0.03 | 0.78 ± 0.04 | 0.86 ± 0.01 |\n"
                "| Dog: Trot | 0.86 ± 0.01 | 0.86 ± 0.01 | 0.86 ± 0.03 |\n"
                "| Dog: Turn | 0.75 ± 0.02 | 0.75 ± 0.03 | 0.82 ± 0.03 |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p016::table::001",
            "page": 16,
            "table_id": "AWR Hyperparameters (Section 5 / Appendix C)",
            "bbox": [48, 120, 560, 420],
            "units": "Hyperparameter settings",
            "text": (
                "[AWR Hyperparameters | Page 16] AWR training hyperparameters: discount factor gamma = 0.99, "
                "TD(lambda) parameter lambda = 0.95, replay buffer capacity |D| = 50,000 (50k default), "
                "temperature beta = 0.05, weight clip = 20, policy network 2 hidden layers of 128->64 ReLU units, "
                "value batch size 256, policy batch size 256."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p025::table::001",
            "page": 25,
            "table_id": "Harness VLA Table 1",
            "bbox": [52, 515, 558, 745],
            "units": "Primitive name, type (Composite / Atomic / VLA), and role",
            "text": (
                "[Harness VLA Table 1 | Page 25] Table 1: Primitive vocabulary (6 shared analytic primitives + 1 VLA primitive + 2 RoboCasa365 mobile-base primitives).\n"
                "| Primitive | Type | Role |\n"
                "|---|---|---|\n"
                "| MOVE_TO | Composite | Move an end-effector to a world-frame Cartesian target using the environment's embedded solver. |\n"
                "| MOVE_POSE | Composite | Move the end-effector while co-varying pose variables such as pitch for reach-limited configurations. |\n"
                "| ROTATE_WRIST | Atomic | Apply a wrist-yaw set-point while holding the current spatial position. |\n"
                "| ROTATE_PITCH | Atomic | Apply a wrist-pitch set-point while holding the current spatial position. |\n"
                "| SET_GRIPPER | Atomic | Drive the gripper to an open or closed set-point for a fixed number of steps. |\n"
                "| RELEASE | Atomic | Open the gripper under a release post-condition. |\n"
                "| VLA_ACT | VLA | Execute a frozen VLA in short bursts for local contact-rich interaction. |\n"
                "| NAVIGATE_TO | Composite (RoboCasa365) | Drive the mobile base to a world-frame location for kitchen-scale staging. |\n"
                "| MOVE_BASE | Atomic (RoboCasa365) | Apply an open-loop local base-velocity set-point for fine repositioning. |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p027::table::001",
            "page": 27,
            "table_id": "Harness VLA Table 2 & Table 3",
            "bbox": [52, 320, 558, 745],
            "units": "Success rate (%) across 100 trials per cell (10 tasks x 10 seeds)",
            "text": (
                "[Harness VLA Table 2 & Table 3 | Page 27] Table 2: Success rate (%) on standard LIBERO (100 trials/suite):\n"
                "| Method | Spatial | Object | Goal | LIBERO-10 | Overall |\n"
                "|---|---|---|---|---|---|\n"
                "| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |\n"
                "| NORA | 85.6 | 89.4 | 80.0 | 63.0 | 79.5 |\n"
                "| pi_0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |\n"
                "| pi_RLinf | 99.0 | 96.0 | 97.0 | 89.0 | 95.3 |\n"
                "| AtomVLA | 96.4 | 99.6 | 97.6 | 94.4 | 97.0 |\n"
                "| Harness VLA (CC) | 97.0 | 100.0 | 94.0 | 93.0 | 96.0 |\n\n"
                "Table 3: Aggregate LIBERO-Pro success rates (%) under instruction-redirection (T) and position-swap (S) perturbations:\n"
                "| Method | Spat-T | Spat-S | Obj-T | Obj-S | Goal-T | Goal-S | L10-T | L10-S | Overall |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n"
                "| OpenVLA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |\n"
                "| pi_0 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 |\n"
                "| pi_0.5 | 1.0 | 20.0 | 1.0 | 17.0 | 2.0 | 38.0 | 1.0 | 8.0 | 11.0 |\n"
                "| MolmoAct | 0.0 | 0.0 | 0.0 | 6.0 | 0.0 | 0.0 | 6.0 | 0.0 | 1.5 |\n"
                "| NORA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |\n"
                "| X-VLA | 0.0 | 0.0 | 8.0 | 2.0 | 9.0 | 1.0 | 10.0 | 0.0 | 3.8 |\n"
                "| AtomVLA | 1.0 | 16.0 | 0.0 | 10.0 | 11.0 | 2.0 | 9.0 | 1.0 | 6.3 |\n"
                "| Cap-X | 14.0 | 12.0 | 18.0 | 22.0 | 17.0 | 26.0 | / | / | 18.2 |\n"
                "| RATS | 31.0 | 29.0 | 63.0 | 61.0 | 36.0 | 43.0 | / | / | 43.8 |\n"
                "| pi_RLinf | 42.0 | 59.0 | 71.0 | 78.0 | 45.0 | 42.0 | 49.0 | 14.0 | 50.0 |\n"
                "| Harness VLA (Codex) | 81.0 | 69.0 | 94.0 | 91.0 | 75.0 | 66.0 | 52.0 | 49.0 | 72.1 |\n"
                "| Harness VLA (CC) | 94.0 | 80.0 | 88.0 | 90.0 | 87.0 | 87.0 | 71.0 | 62.0 | 82.4 |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p028::table::001",
            "page": 28,
            "table_id": "Harness VLA Table 4",
            "bbox": [52, 260, 558, 430],
            "units": "Success rate (%) across 250 rollouts (50 tasks x 5 seeds)",
            "text": (
                "[Harness VLA Table 4 | Page 28] Table 4: RoboCasa365 success rates (%) on Atomic-Seen, Composite-Seen, and Composite-Unseen (250 total rollouts):\n"
                "| Method | Atomic-Seen | Composite-Seen | Composite-Unseen | Overall |\n"
                "|---|---|---|---|---|\n"
                "| RLDX-1 | 60.0 | 21.3 | 5.0 | 30.0 |\n"
                "| WorldDreamer | 66.3 | 26.7 | 9.0 | 35.3 |\n"
                "| pi_0.5 | 39.6 | 7.1 | 1.2 | 16.9 |\n"
                "| pi_0 | 34.6 | 6.1 | 1.1 | 14.8 |\n"
                "| Harness VLA (Codex) | 92.2 | 61.3 | 13.8 | 57.2 (text notes 57.1%) |\n"
                "| Harness VLA (CC) | 80.0 | 47.5 | 15.0 | 48.8 (text notes 48.6%) |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p029::table::001",
            "page": 29,
            "table_id": "Harness VLA Table 5 & Table 6",
            "bbox": [52, 410, 558, 745],
            "units": "Success rate (%)",
            "text": (
                "[Harness VLA Table 5 & Table 6 | Page 29] Table 5: Per-task success rate (%) on LIBERO-PRO GOAL: zero-shot Harness VLA (CC) (no Task Specific Memory retrieval, 10 seeds/task) vs Cap-X:\n"
                "- Pos (S) Average: Cap-X = 25.6%, Zero-shot Harness VLA (CC) = 31.0% (compared to 87.0% few-shot in Table 3; Goal-S position-swap drops by 87.0% - 31.0% = 56.0 percentage points).\n"
                "- Task (T) Average: Cap-X = 16.8%, Zero-shot Harness VLA (CC) = 79.0% (compared to 87.0% few-shot in Table 3; Goal-T instruction-redirection drops by 87.0% - 79.0% = 8.0 percentage points).\n\n"
                "Table 6: RoboTwin C2R success rates (%) across 50 tasks x 5 randomized seeds (250 rollouts):\n"
                "| Benchmark | GR00T-N1.7 | pi_0.5 | StarVLA | LingBot-VLA | Harness VLA (Codex) | Harness VLA (CC) |\n"
                "|---|---|---|---|---|---|---|\n"
                "| RoboTwin C2R | 20.7% | 47.9% | 10.6% | 50.4% | 58.0% | 58.4% |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p035::table::001",
            "page": 35,
            "table_id": "Harness VLA Table 7",
            "bbox": [52, 570, 558, 745],
            "units": "Success rate (%) and Mean Execution Time (seconds)",
            "text": (
                "[Harness VLA Table 7 | Page 35] Table 7: Success rates (%) and execution time (s) on LIBERO-Pro from the RPent Flash comparison (800 episodes total; 78 of 80 task identities have compiled Flash cards since goal_swap_t0 and 10_swap_t9 have no successful source trace; 581/800 episodes succeed in Flash Mode (max) = 72.63%; speedup vs Codex reasoning high: 529.23s / 56.17s = 9.42x; vs Codex no reasoning: 344.17s / 56.17s = 6.13x):\n"
                "| Method | Spat-T | Spat-S | Obj-T | Obj-S | Goal-T | Goal-S | L10-T | L10-S | Overall (%) | Time (s) |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n"
                "| Codex (reasoning high) | 88.0 | 88.0 | 95.0 | 98.0 | 77.0 | 56.0 | 68.0 | 58.0 | 78.50 | 529.23 |\n"
                "| Codex (no reasoning) | 86.0 | 67.0 | 94.0 | 92.0 | 15.0 | 64.0 | 43.0 | 39.0 | 62.50 | 344.17 |\n"
                "| Flash Mode (max) | 79.0 | 70.0 | 86.0 | 93.0 | 74.0 | 65.0 | 60.0 | 54.0 | 72.63 | 56.17 |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p046::table::001",
            "page": 46,
            "table_id": "Harness VLA Table 9",
            "bbox": [52, 520, 558, 745],
            "units": "Primitive availability across benchmarks",
            "text": (
                "[Harness VLA Table 9 | Page 46] Table 9: Primitive availability across the three benchmark embodiments:\n"
                "| Primitive | LIBERO | RoboCasa365 | RoboTwin C2R |\n"
                "|---|---|---|---|\n"
                "| MOVE_TO | yes | yes | yes |\n"
                "| MOVE_POSE | yes | via composition | - |\n"
                "| ROTATE_WRIST | yes | - | yes |\n"
                "| ROTATE_PITCH | yes | yes | - |\n"
                "| SET_GRIPPER | yes | yes | yes |\n"
                "| RELEASE | yes | yes | yes |\n"
                "| VLA_ACT | yes | yes | yes |\n"
                "| NAVIGATE_TO | - | yes | - |\n"
                "| MOVE_BASE | - | yes | - |\n"
                "| arm binding | - | - | yes |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p050::table::001",
            "page": 50,
            "table_id": "Harness VLA Table 10, 11, 12, 14",
            "bbox": [52, 480, 558, 700],
            "units": "Tasks, evaluation seeds per task, and reported rollouts",
            "text": (
                "[Harness VLA Table 14 | Page 50] Table 14: Summary of evaluation benchmarks (combining Tables 10, 11, 12, 13):\n"
                "| Benchmark | Tasks | Trials/task (Eval seeds) | Reported rollouts |\n"
                "|---|---|---|---|\n"
                "| LIBERO | 40 (4 suites x 10 tasks) | 10 (s1-s10; seed 0 for memory) | 400 |\n"
                "| LIBERO-Pro | 80 (8 cells x 10 tasks) | 10 (s1-s10; seed 0 for memory) | 800 |\n"
                "| RoboCasa365 | 50 (18 Atomic-Seen, 16 Composite-Seen, 16 Composite-Unseen) | 5 (s1-s5; seed 0 for memory) | 250 |\n"
                "| RoboTwin C2R | 50 | 5 (demo_randomized seeds; demo_clean for memory) | 250 |\n"
                "Total reported rollouts across all 4 benchmarks = 400 + 800 + 250 + 250 = 1700 rollouts."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p053::table::001",
            "page": 53,
            "table_id": "Harness VLA Table 15",
            "bbox": [52, 460, 558, 745],
            "units": "Post-training hyperparameters",
            "text": (
                "[Harness VLA Table 15 | Page 53] Table 15: Post-training configuration of LingBot-VLA on RoboTwin:\n"
                "- Optimizer: AdamW | Learning rate: 1 x 10^-4 | Vision encoder LR: 1 x 10^-6 | Weight decay: 0 | Loss function: L1 Flow Matching (L1 FM)\n"
                "- Sequence Modeling: Chunk size = 50 | Max sequence length = 2048 | Flow steps = 10 | Max action dimension = 75 | Max state dimension = 75\n"
                "- Training Setup: Global batch size = 256 | Image resolution = 224 x 224 | Camera views = top + wrist_left + wrist_right\n"
                "- System: Precision = mixed precision (bf16/fp32) | Distributed training = FSDP2"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p059::table::001",
            "page": 59,
            "table_id": "Harness VLA Table 19 & Table 20",
            "bbox": [52, 120, 558, 380],
            "units": "Call count and percentage (%) of manipulation-primitive calls",
            "text": (
                "[Harness VLA Table 19 & Table 20 | Pages 59-60] Table 19: Canonical primitive usage across benchmark environments (count and percentage):\n"
                "| Canonical primitive | Kind | LIBERO | RoboTwin C2R | RoboCasa365 |\n"
                "|---|---|---|---|---|\n"
                "| MOVE_TO | Analytic composite | 6263 (61.8%) | 685 (40.9%) | 3004 (38.7%) |\n"
                "| MOVE_POSE | Analytic composite | 203 (2.0%) | - | - |\n"
                "| NAVIGATE_TO | Analytic composite | - | - | 701 (9.0%) |\n"
                "| ROTATE_WRIST | Analytic atomic | 44 (0.4%) | 1 (0.1%) | - |\n"
                "| ROTATE_PITCH | Analytic atomic | 58 (0.6%) | - | 66 (0.8%) |\n"
                "| SET_GRIPPER | Analytic atomic | 1137 (11.2%) | 71 (4.2%) | 371 (4.8%) |\n"
                "| RELEASE | Analytic atomic | 831 (8.2%) | 124 (7.4%) | 76 (1.0%) |\n"
                "| MOVE_BASE | Analytic atomic | - | - | 808 (10.4%) |\n"
                "| VLA_ACT | VLA | 1598 (15.8%) | 794 (47.4%) | 2746 (35.3%) |\n"
                "| Total | All | 10134 (100.0%) | 1675 (100.0%) | 7772 (100.0%) |\n\n"
                "Table 20 (Page 60): Primitive usage grouped by class (percentage of total manipulation calls):\n"
                "- Analytic primitives: LIBERO = 8536 (84.2%), RoboTwin C2R = 881 (52.6%), RoboCasa365 = 5026 (64.7%).\n"
                "- VLA primitive (VLA_ACT): LIBERO = 1598 (15.8%), RoboTwin C2R = 794 (47.4%), RoboCasa365 = 2746 (35.3%). "
                "(Thus VLA_ACT accounts for 47.4% of total manipulation calls on RoboTwin C2R and 35.3% of total manipulation calls on RoboCasa365 / Atomic-Seen.)"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p061::table::001",
            "page": 61,
            "table_id": "Harness VLA Table 21 & Table 22",
            "bbox": [52, 220, 558, 560],
            "units": "Success rate (%)",
            "text": (
                "[Harness VLA Table 21 & Table 22 | Page 61] Additional Planner Results with GPT-6 Astra (Total rollouts across LIBERO-Pro [800] + RoboCasa365 [250] = 1050 rollouts; total successful rollouts for GPT-6 Astra = 741 + 148 = 889 successful rollouts; Composite-Unseen gain over GPT-5.5 = 42.50% - 13.8% = +28.7 percentage points):\n"
                "Table 21: LIBERO-Pro success rates (%) with GPT-6 Astra (741/800 successes = 92.63% overall):\n"
                "| Planner | Spat-T | Spat-S | Obj-T | Obj-S | Goal-T | Goal-S | L10-T | L10-S | Overall |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n"
                "| Codex (GPT-5.5) | 81.0 | 69.0 | 94.0 | 91.0 | 75.0 | 66.0 | 52.0 | 49.0 | 72.1 |\n"
                "| Claude Code (Opus-4.7) | 94.0 | 80.0 | 88.0 | 90.0 | 87.0 | 87.0 | 71.0 | 62.0 | 82.4 |\n"
                "| Codex (GPT-6 Astra) | 100.0 | 98.0 | 100.0 | 99.0 | 88.0 | 99.0 | 85.0 | 72.0 | 92.63 |\n\n"
                "Table 22: RoboCasa365 success rates (%) with GPT-6 Astra (148/250 successes = 59.20% overall):\n"
                "| Planner | Atomic-Seen | Composite-Seen | Composite-Unseen | Overall |\n"
                "|---|---|---|---|---|\n"
                "| Codex (GPT-5.5) | 92.2 | 61.3 | 13.8 | 57.2 |\n"
                "| Claude Code (Opus-4.7) | 80.0 | 47.5 | 15.0 | 48.8 |\n"
                "| Codex (GPT-6 Astra) | 87.78 | 43.75 | 42.50 | 59.20 |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p070::table::001",
            "page": 70,
            "table_id": "AMP Table 1 & Table 2",
            "bbox": [48, 220, 558, 720],
            "units": "Normalized task return in [0, 1] (Table 1); Dataset size in seconds (s), clips, and subjects (Table 2)",
            "text": (
                "[AMP Table 1 & Table 2 | Page 70] Table 1: Performance statistics of combining AMP with additional task objectives (average normalized task return in [0, 1] ± SD across 3 seeds x 32 episodes):\n"
                "| Character | Task | Dataset | Task Return |\n"
                "|---|---|---|---|\n"
                "| Humanoid | Target Heading | Locomotion | 0.90 ± 0.01 |\n"
                "| Humanoid | Target Heading | Walk | 0.46 ± 0.01 |\n"
                "| Humanoid | Target Heading | Run | 0.63 ± 0.01 |\n"
                "| Humanoid | Target Heading | Stealthy | 0.89 ± 0.02 |\n"
                "| Humanoid | Target Heading | Zombie | 0.94 ± 0.00 |\n"
                "| Humanoid | Target Location | Locomotion | 0.63 ± 0.01 |\n"
                "| Humanoid | Target Location | Zombie | 0.50 ± 0.00 |\n"
                "| Humanoid | Obstacles | Run + Leap + Roll | 0.27 ± 0.10 |\n"
                "| Humanoid | Stepping Stones | Cartwheel | 0.43 ± 0.03 |\n"
                "| Humanoid | Stepping Stones | Jump | 0.56 ± 0.12 |\n"
                "| Humanoid | Dribble | Locomotion | 0.78 ± 0.05 |\n"
                "| Humanoid | Dribble | Zombie | 0.60 ± 0.04 |\n"
                "| Humanoid | Strike | Walk + Punch | 0.73 ± 0.02 |\n"
                "| T-Rex | Target Location | Locomotion | 0.36 ± 0.03 |\n\n"
                "Table 2: Summary statistics of the different datasets used to train the motion priors (Note: Humanoid Locomotion is 434.1s vs Zombie 18.3s, so Locomotion is 434.1 / 18.3 = 23.72 times larger in duration, while Zombie achieves 0.94 ± 0.00 vs Locomotion 0.90 ± 0.01 on Target Heading):\n"
                "| Character | Dataset | Size (s) | Clips | Subjects |\n"
                "|---|---|---|---|---|\n"
                "| Humanoid | Cartwheel | 13.6 | 3 | 1 |\n"
                "| Humanoid | Jump | 28.6 | 10 | 4 |\n"
                "| Humanoid | Locomotion | 434.1 | 56 | 8 |\n"
                "| Humanoid | Run | 204.4 | 47 | 3 |\n"
                "| Humanoid | Run + Leap + Roll | 22.1 | 10 | 7 |\n"
                "| Humanoid | Stealthy | 136.5 | 3 | 1 |\n"
                "| Humanoid | Walk | 229.6 | 9 | 5 |\n"
                "| Humanoid | Walk + Punch | 247.8 | 15 | 9 |\n"
                "| Humanoid | Zombie | 18.3 | 1 | 1 |\n"
                "| T-Rex | Locomotion | 10.5 | 5 | 1 |"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p073::table::001",
            "page": 73,
            "table_id": "AMP Table 3",
            "bbox": [48, 350, 300, 720],
            "units": "Dataset Size in seconds (s); Average DTW-aligned pose error in meters (m) ± SD",
            "text": (
                "[AMP Table 3 | Page 73] Table 3: Performance statistics of imitating individual motion clips without task objectives "
                "(average DTW-aligned pose error in meters ± SD across 3 seeds x 32 episodes; lower is better):\n"
                "| Character | Motion | Dataset Size (s) | Motion Tracking (m) | AMP (Ours) (m) |\n"
                "|---|---|---|---|---|\n"
                "| Humanoid | Back-Flip | 1.75s | 0.076 ± 0.021 | 0.150 ± 0.028 |\n"
                "| Humanoid | Cartwheel | 2.72s | 0.039 ± 0.011 | 0.067 ± 0.014 |\n"
                "| Humanoid | Crawl | 2.93s | 0.044 ± 0.001 | 0.049 ± 0.007 |\n"
                "| Humanoid | Dance | 1.62s | 0.038 ± 0.001 | 0.055 ± 0.015 |\n"
                "| Humanoid | Front-Flip | 1.65s | 0.278 ± 0.040 | 0.425 ± 0.010 |\n"
                "| Humanoid | Jog | 0.83s | 0.029 ± 0.001 | 0.056 ± 0.001 |\n"
                "| Humanoid | Jump | 1.77s | 0.033 ± 0.001 | 0.083 ± 0.022 |\n"
                "| Humanoid | Roll | 2.02s | 0.072 ± 0.018 | 0.088 ± 0.008 |\n"
                "| Humanoid | Run | 0.80s | 0.028 ± 0.002 | 0.075 ± 0.015 |\n"
                "| Humanoid | Spin | 1.00s | 0.063 ± 0.022 | 0.047 ± 0.002 |\n"
                "| Humanoid | Side-Flip | 2.44s | 0.191 ± 0.043 | 0.124 ± 0.012 |\n"
                "| Humanoid | Spin-Kick | 1.28s | 0.042 ± 0.001 | 0.058 ± 0.012 |\n"
                "| Humanoid | Walk | 1.30s | 0.018 ± 0.005 | 0.030 ± 0.001 |\n"
                "| Humanoid | Zombie | 1.68s | 0.049 ± 0.013 | 0.058 ± 0.014 |\n"
                "| T-Rex | Turn | 2.13s | 0.098 ± 0.011 | 0.284 ± 0.023 |\n"
                "| T-Rex | Walk | 2.00s | 0.069 ± 0.005 | 0.096 ± 0.027 |\n"
                "| Dog | Canter | 0.45s | 0.026 ± 0.002 | 0.034 ± 0.002 |\n"
                "| Dog | Pace | 0.63s | 0.020 ± 0.001 | 0.024 ± 0.003 |\n"
                "| Dog | Spin | 0.73s | 0.026 ± 0.002 | 0.086 ± 0.008 |\n"
                "| Dog | Trot | 0.52s | 0.019 ± 0.001 | 0.026 ± 0.001 |\n"
                "Note: AMP achieves lower pose error than Motion Tracking on Humanoid Spin (0.047m vs 0.063m) and Humanoid Side-Flip (0.124m vs 0.191m)."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p078::table::001",
            "page": 78,
            "table_id": "AMP Table 4 & Table 5",
            "bbox": [48, 220, 558, 720],
            "units": "Hyperparameter values",
            "text": (
                "[AMP Table 4 & Table 5 | Page 78] Table 4: AMP hyperparameters:\n"
                "- Task-Reward Weight w_G = 0.5 | Style-Reward Weight w_S = 0.5 | Gradient Penalty w_gp = 10\n"
                "- Samples Per Update Iteration = 4096 | Batch Size = 256 | Discriminator Batch Size K = 256\n"
                "- Policy Stepsize (Single-Clip Imitation) = 2 x 10^-6 | Policy Stepsize (Tasks) = 4 x 10^-6\n"
                "- Value Stepsize (Single-Clip Imitation) = 10^-4 | Value Stepsize (Tasks) = 2 x 10^-5\n"
                "- Discriminator Stepsize = 10^-5 | Discriminator Replay Buffer Size B = 10^5\n"
                "- Discount gamma (Single-Clip Imitation) = 0.95 | Discount gamma (Tasks) = 0.99\n"
                "- SGD Momentum = 0.9 | GAE(lambda) = 0.95 | TD(lambda) = 0.95 | PPO Clip Threshold = 0.02\n\n"
                "Table 5: Latent space model hyperparameters:\n"
                "- Latent Encoding Dimension = 16 | KL-Regularizer lambda = 10^-4 | Samples Per Update Iteration = 4096 | Batch Size = 256\n"
                "- Policy pi Stepsize (Pre-Training) = 2.5 x 10^-6 | High-level u Policy Stepsize (Downstream Task) = 10^-4 | Value Stepsize = 10^-3\n"
                "- Discount gamma (Pre-Training) = 0.95 | Discount gamma (Downstream Task) = 0.99 | PPO Clip Threshold = 0.02"
            ),
        },
        {
            "id": f"{SOURCE_ID}::p079::table::001",
            "page": 79,
            "table_id": "AMP Table 6",
            "bbox": [300, 480, 558, 660],
            "units": "Dataset size in seconds (s); Normalized Heading Return and Waving Return in [0, 1] ± SD",
            "text": (
                "[AMP Table 6 | Page 79] Table 6: Performance of policies trained using different datasets on the spatial compositional task "
                "(simultaneously following a target heading and waving the character's right hand at a target height, averaged across 100 episodes):\n"
                "| Dataset (Size) | Heading Return | Waving Return |\n"
                "|---|---|---|\n"
                "| Wave (51.7s) | 0.683 ± 0.195 | 0.949 ± 0.144 |\n"
                "| Walk (229.7s) | 0.945 ± 0.192 | 0.306 ± 0.378 |\n"
                "| Wave + Walk (281.4s) | 0.885 ± 0.184 | 0.891 ± 0.202 |"
            ),
        },
    ]


def _visual_items() -> list[dict[str, Any]]:
    """Curated, visually verified figure/diagram and chart chunks linked to cropped images in data/advanced/crops/."""
    return [
        # --- FIGURES & DIAGRAMS (modality="figure_diagram") ---
        {
            "id": f"{SOURCE_ID}::p009::figure::001",
            "page": 9,
            "modality": "figure_diagram",
            "figure_id": "AWR Figure 5",
            "crop_files": [
                "data/advanced/crops/p009_img05_Im15.png",
                "data/advanced/crops/p009_img06_Im16.png",
                "data/advanced/crops/p009_img07_Im17.png",
                "data/advanced/crops/p009_img08_Im18.png",
            ],
            "bbox": [48, 220, 558, 490],
            "text": (
                "[AWR Figure 5 | Page 9 | Crops: p009_img05_Im15.png .. p009_img08_Im18.png] "
                "Figure 5: Time-lapse snapshots of a 34 DoF (degrees of freedom) humanoid and an 82 DoF dog trained with AWR "
                "to imitate reference motion capture clips from real-world subjects. The four subfigures are: "
                "(a) Humanoid: Cartwheel, (b) Humanoid: Spinkick, (c) Dog: Trot, and (d) Dog: Turn."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p020::figure::001",
            "page": 20,
            "modality": "figure_diagram",
            "figure_id": "Harness VLA Figure 1",
            "crop_files": ["data/advanced/crops/p020_img02_Im2.png"],
            "bbox": [48, 190, 560, 580],
            "text": (
                "[Harness VLA Figure 1 | Page 20 | Crop: p020_img02_Im2.png] "
                "Figure 1: Harness VLA system overview diagram (2856x1770 px). "
                "Left input column shows three inputs to the Agentic Planner: "
                "1) Task Description ('Pick up the book and place it in the back compartment of the caddy'), "
                "2) Observation (RGB and depth camera images), and "
                "3) Robot State: end-effector position eef_pos [x,y,z], end-effector quaternion eef_quat [x,y,z,w], and gripper joint position gripper_qpos [q1,q2]. "
                "Center shows the Agentic Planner (Opus 4.7 / GPT 5.5) interacting bidirectionally with the Primitive Library (top: VLA Primitive VLA_ACT and Analytic Primitives MOVE_TO, ROTATE, SET_GRIPPER) "
                "and Memory (right: Task Specific Memory storing structured step-by-step primitive calls such as VLA_ACT prompt and MOVE_TO position (x,y,z) used via few-shot arrow; Global Memory storing generalized heuristics extracted from exploration). "
                "Middle horizontal command strip shows the 7-step execution sequence: Initial State -> VLA_ACT -> MOVE_TO -> SET_GRIPPER -> VLA_ACT -> MOVE_TO -> MOVE_TO. "
                "Bottom row shows 5 real-world dual-Franka robot tasks: Basket Placement (Frozen VLA) vs Target Redirection, Visual Sorting, Grasp Recovery, and Ordered Placement (HarnessVLA). "
                "Right column bar charts summarize benchmark gains: LIBERO Pro (pi_0.5 11.0%, Cap-X 18.2%, RATS 43.8%, Ours 82.4% [+39%, 1.9x]), RoboCasa365 (pi_0.5 16.0%, RLDX-1 30.0%, Ours 55.4% [+25%]), and RoboTwin C2R (StarVLA 10.6%, pi_0.5 47.9%, Ours 58.4% [+10%])."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p022::figure::001",
            "page": 22,
            "modality": "figure_diagram",
            "figure_id": "Harness VLA Figure 2",
            "crop_files": ["data/advanced/crops/p022_img01_Im3.png"],
            "bbox": [52, 340, 558, 595],
            "text": (
                "[Harness VLA Figure 2 | Page 22 | Crop: p022_img01_Im3.png] "
                "Figure 2: Conceptual diagram 'Primitive composition extends a frozen VLA beyond its trajectory distribution' (2690x1328 px). "
                "The task description box on the left says: 'Pick the pan and the sponge and place them into the sink. Then turn on the water.' "
                "The light-blue outer region represents 'Deployment Perturbations' (labeled with 'Semantic Retargeting' and 'Spatial Shift'), while the three orange-hatched ellipses represent 'VLA in Distribution Trajectories'. "
                "Red dashed arrows with red X marks show 'VLA Unstable Rollout' failing when trying to bridge across perturbed out-of-distribution space directly. "
                "The solid gray curve shows 'HarnessVLA Rollout': green circles ('Analytic Primitive Control') perceive the target ('Perceive Target'), re-ground bindings, and transport the robot between regions, while orange circles ('VLA Contact-Rich Control') invoke VLA_ACT only inside the in-distribution local contact regions."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p029::figure::001",
            "page": 29,
            "modality": "figure_diagram",
            "figure_id": "Harness VLA Figure 3",
            "crop_files": [
                "data/advanced/crops/p029_img01_Im4.jpg",
                "data/advanced/crops/p029_img02_Im5.jpg",
                "data/advanced/crops/p029_img03_Im6.jpg",
            ],
            "bbox": [52, 220, 558, 380],
            "text": (
                "[Harness VLA Figure 3 | Page 29 | Crops: p029_img01_Im4.jpg .. p029_img06_Im9.jpg] "
                "Figure 3: Terminal-state frames for two LIBERO-Pro cells comparing standard pi_RLinf vs Harness VLA. "
                "First triplet (Obj Std, Obj-T pi_RLinf, Obj-T Ours): under instruction-redirection (Obj-T), pi_RLinf repeats the standard training behavior and grasps the wrong object, whereas Harness VLA re-grounds the new target instruction and succeeds. "
                "Second triplet (Goal Std, Goal-S pi_RLinf, Goal-S Ours): under position-swap (Goal-S), pi_RLinf blindly moves the object toward the old training-time region after layout swap, whereas Harness VLA re-grounds the swapped goal position via the planner and analytic staging."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p035::figure::001",
            "page": 35,
            "modality": "figure_diagram",
            "figure_id": "Harness VLA Figure 9",
            "crop_files": [
                "data/advanced/crops/p035_img01_Im53.jpg",
                "data/advanced/crops/p035_img04_Im56.jpg",
                "data/advanced/crops/p035_img07_Im59.jpg",
            ],
            "bbox": [52, 210, 558, 490],
            "text": (
                "[Harness VLA Figure 9 | Page 35 | Crops: p035_img01_Im53.jpg .. p035_img09_Im61.jpg] "
                "Figure 9: 'From the learned routine to Flash composition' on a physical bimanual robot (3 rows progressing left to right): "
                "1) Top row (VLA fullshot): grasp the container, hand it off between arms, and place it in the bin. "
                "2) Middle row (Harness VLA Flash Mode): hold the container, acquire the receiving bowl, and tilt the container to pour steel beads. "
                "3) Bottom row (Harness VLA Flash Mode): acquire the sponge while holding the plate, wipe the plate, and stow the plate after setting down the sponge. Both Flash rows execute offline compiled plans without online LLM planner calls."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p065::figure::001",
            "page": 65,
            "modality": "figure_diagram",
            "figure_id": "AMP Figure 2",
            "crop_files": ["data/advanced/crops/p065_img01_Im3.png"],
            "bbox": [304, 515, 558, 720],
            "text": (
                "[AMP Fig. 2 | Page 65 | Crop: p065_img01_Im3.png] "
                "Fig. 2: Schematic overview of the Adversarial Motion Priors (AMP) system (1149x684 px). "
                "The diagram has 5 main blocks and signal arrows: "
                "1) 'Dataset' (reference motion clips) and 'Environment' (sending state s_t) both feed into the 'Motion Prior' block (red box showing adversarial discriminator separating blue 'reference' vs red 'simulation' distributions). "
                "2) 'Motion Prior' outputs the style-reward r_t^S. "
                "3) 'Task' block outputs two signals: the task-reward r_t^G and the task goal g. "
                "4) An addition node (+) sums the style-reward r_t^S and task-reward r_t^G to form the combined scalar reward r_t = r_t^S + r_t^G. "
                "5) The 'Policy' neural network block receives three upward input arrows: state s_t (from Environment), combined reward r_t (from the + node for policy training), and goal g (from Task), and outputs action a_t back to the 'Environment'."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p072::figure::001",
            "page": 72,
            "modality": "figure_diagram",
            "figure_id": "AMP Figure 7",
            "crop_files": [
                "data/advanced/crops/p072_img01_Im25.png",
                "data/advanced/crops/p072_img02_Im26.png",
                "data/advanced/crops/p072_img03_Im27.png",
            ],
            "bbox": [48, 560, 558, 735],
            "text": (
                "[AMP Fig. 7 | Page 72 | Crops: p072_img01_Im25.png .. p072_img03_Im27.png] "
                "Fig. 7: Non-humanoid simulated characters trained with AMP: a 59 DoF (degrees of freedom) T-Rex and a 64 DoF dog. "
                "The three subfigures show: (a) T-Rex (Walk), (b) Dog (Trot), and (c) Dog (Canter)."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p085::figure::001",
            "page": 85,
            "modality": "figure_diagram",
            "figure_id": "FBRL Figure 1",
            "crop_files": [
                "data/advanced/crops/p085_img01_Im1.png",
                "data/advanced/crops/p085_img02_Im2.png",
            ],
            "bbox": [120, 320, 490, 470],
            "text": (
                "[FBRL Figure 1 | Page 85 | Crops: p085_img01_Im1.png, p085_img02_Im2.png] "
                "Figure 1: Gridworld and Towers of Hanoi environments used in Forward-Backward Reinforcement Learning (FBRL). "
                "Left diagram (p085_img01_Im1.png, 244x244 px): A 3x3 Gridworld where the robot icon is initialized in the bottom-left cell (start state) and the text 'Goal' is in the top-right cell (reward = +1 at top-right Goal, step cost = -0.01 per time-step). "
                "Right diagram (p085_img02_Im2.png, 353x243 px): A 3-pillar Towers of Hanoi environment with 2 discs: the smaller pink disc is on the 1st (leftmost) brown pillar on top of the larger green-teal disc, wait—in the image p085_img02_Im2.png both the small pink disc and large teal disc are stacked on the 1st (leftmost) pillar while pillars 2 and 3 are empty, and page 86 text notes that bit-string state representation has 3 bits per disc indicating which of the 3 pillars each disc is on (e.g., [1, 0, 0, 1, 0, 0] when the small disc is on the first pillar and the large disc is on the first/third pillar)."
            ),
        },
        # --- CHARTS (modality="chart") ---
        {
            "id": f"{SOURCE_ID}::p008::chart::001",
            "page": 8,
            "modality": "chart",
            "figure_id": "AWR Figure 3",
            "crop_files": [
                "data/advanced/crops/p008_img01_Im10.png",
                "data/advanced/crops/p008_img02_Im6.png",
                "data/advanced/crops/p008_img03_Im7.png",
                "data/advanced/crops/p008_img04_Im8.png",
                "data/advanced/crops/p008_img05_Im9.png",
            ],
            "bbox": [48, 595, 558, 745],
            "text": (
                "[AWR Figure 3 | Page 8 | Crops: p008_img01_Im10.png .. p008_img05_Im9.png] "
                "Figure 3: Learning curves comparing 7 algorithms (legend in p008_img01_Im10.png: TRPO [blue], PPO [orange], DDPG [purple], TD3 [brown], SAC [pink], RWR [green], AWR (Ours) [red]) on OpenAI Gym tasks. "
                "X-axis: Samples (up to 2.0 x 10^7 samples for HalfCheetah-v2). Y-axis: Return. "
                "In HalfCheetah-v2 (p008_img02_Im6.png), DDPG (purple) reaches the highest curve above 10,000 (~10,563), SAC (pink) and AWR (red) both converge tightly around ~9,100–9,300, PPO (orange) reaches ~4,920, TD3 (brown) ~4,309, TRPO (blue) ~3,302, and RWR (green) stays lowest around ~1,400."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p009::chart::001",
            "page": 9,
            "modality": "chart",
            "figure_id": "AWR Figure 4",
            "crop_files": [
                "data/advanced/crops/p009_img01_Im11.png",
                "data/advanced/crops/p009_img02_Im12.png",
                "data/advanced/crops/p009_img03_Im13.png",
                "data/advanced/crops/p009_img04_Im14.png",
            ],
            "bbox": [48, 520, 558, 745],
            "text": (
                "[AWR Figure 4 | Page 9 | Crops: p009_img01_Im11.png .. p009_img04_Im14.png] "
                "Figure 4: Ablation and Replay Buffer capacity learning curves on HalfCheetah-v2 and Walker2d-v2 (X-axis: Samples from 0.0 to 2.0 x 10^7; Y-axis: Return). "
                "1) Left panel (Ablation: HalfCheetah-v2, p009_img01_Im11.png): compares 5 curves: 'AWR (Ours)' (red) achieves highest return (~9,100 at 2.0x10^7 samples), followed by 'No TD(lambda)' (green, ~5,900), 'On-Policy' (blue, ~4,500), 'No Baseline' (orange, ~3,600), and 'RWR' (purple, lowest at ~1,400). In Ablation: Walker2d-v2 (p009_img02_Im12.png), 'AWR (Ours)' (red) and 'No TD(lambda)' (green) both reach ~5,800–6,000, 'No Baseline' (orange) reaches ~4,300, 'On-Policy' (blue) fluctuates around ~1,800–2,300, and 'RWR' (purple) collapses below ~500. "
                "2) Right panel (Replay Buffer: HalfCheetah-v2, p009_img03_Im13.png): compares 5 replay buffer capacities: '5k' (blue), '20k' (orange), '50k (Default)' (green), '100k' (red), and '500k' (purple). On HalfCheetah-v2, '50k (Default)' (green) achieves the highest return (~9,100), '100k' (red) and '20k' (orange) reach ~7,900–8,200, '500k' (purple) progresses more slowly to ~7,200, and the smallest buffer '5k' (blue) plateaus lowest around ~4,200–4,500 due to overfitting/instability."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p010::chart::001",
            "page": 10,
            "modality": "chart",
            "figure_id": "AWR Figure 7",
            "crop_files": [
                "data/advanced/crops/p010_img03_Im21.png",
                "data/advanced/crops/p010_img04_Im22.png",
                "data/advanced/crops/p010_img05_Im23.png",
                "data/advanced/crops/p010_img06_Im24.png",
            ],
            "bbox": [48, 430, 558, 600],
            "text": (
                "[AWR Figure 7 | Page 10 | Crops: p010_img03_Im21.png .. p010_img06_Im24.png] "
                "Figure 7: Bar charts of performance on static off-policy datasets (1 million timesteps from SAC demo policies) across 4 OpenAI Gym tasks: Ant-v2, HalfCheetah-v2, Hopper-v2, and Walker2d-v2. "
                "X-axis categories (7 bars): BC (blue), PPO (orange), TD3 (green), SAC (purple), BCQ (brown), BEAR (pink), AWR (red), with a black horizontal dashed line marking 'Demo' policy return. Y-axis: Return. "
                "- Ant-v2 (p010_img03_Im21.png): Demo line is at ~4,400. AWR (red) is the highest bar at ~4,550 (exceeding Demo), BEAR (pink) is ~3,400, BCQ (brown) is ~3,150, BC (blue) is ~2,800, PPO (orange) is ~650, and TD3/SAC are below ~400. "
                "- HalfCheetah-v2 (p010_img04_Im22.png): Demo dashed line is at ~3,950. BEAR (pink) is highest at ~5,020, BCQ (brown) ~4,730, BC (blue) ~4,420, AWR (red) ~4,230 (above Demo), SAC ~4,020, TD3 ~3,860, while PPO (orange) is near 0. "
                "- Hopper-v2 (p010_img05_Im23.png): Demo dashed line is at ~1,025. BEAR (pink) is ~1,610, AWR (red) is second highest at ~1,480 (well above Demo), BCQ (brown) ~1,060, BC (blue) ~1,030, PPO (orange) ~485, and TD3/SAC are near 0. "
                "- Walker2d-v2 (p010_img06_Im24.png): Demo dashed line is at ~2,920. AWR (red) is the highest bar at ~4,480 (far exceeding Demo, BCQ ~3,260, BEAR ~3,080, and BC ~2,880), while PPO, TD3, and SAC have mean bars near 0."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p030::chart::001",
            "page": 30,
            "modality": "chart",
            "figure_id": "Harness VLA Figure 4",
            "crop_files": [
                "data/advanced/crops/p030_img01_Im10.png",
                "data/advanced/crops/p030_img02_Im11.png",
                "data/advanced/crops/p030_img03_Im12.png",
            ],
            "bbox": [52, 52, 558, 260],
            "text": (
                "[Harness VLA Figure 4 | Page 30 | Crops: p030_img01_Im10.png, p030_img02_Im11.png, p030_img03_Im12.png] "
                "Figure 4: 'Adaptive VLA invocation improves success across benchmarks.' Three line charts plotting Y-axis 'Cumulative task success (%)' vs X-axis 'VLA primitive invocations (<= k)':\n"
                "1) Panel (a) LIBERO-PRO (p030_img01_Im10.png): X-axis k from 0 to 6. Blue dashed reference line 'VLA full-shot (50%)'; gray dashed ceiling line 'HarnessVLA, all invocations (82%)'. Data points on the orange curve: k=0 -> 3%, k=1 -> 52%, k=2 -> 68%, k=3 -> 73%, k=4 -> 77%, k=5 -> 79%, k=6 -> 80%. Already at k=1 (52%), Harness VLA surpasses the 50% VLA full-shot baseline.\n"
                "2) Panel (b) ROBOCASA365 (p030_img02_Im11.png): X-axis k from 0 to 10. Blue dashed line 'VLA full-shot (30%)'; gray dashed line 'Harness VLA, all invocations (48%)'. Data points on the orange curve: k=0 -> 0%, k=1 -> 2%, k=2 -> 11%, k=3 -> 21%, k=4 -> 26%, k=5 -> 29%, k=6 -> 31%, k=7 -> 32%, k=8 -> 35%, k=9 -> 36%, k=10 -> 39%. The curve first surpasses the 30% VLA full-shot baseline at k=6 (reaching 31%).\n"
                "3) Panel (c) ROBOTWIN C2R (p030_img03_Im12.png): X-axis k from 0 to 8. Blue dashed line 'VLA full-shot (50%)'; gray dashed line 'Harness VLA, all invocations (58%)'. Data points on the orange curve: k=0 -> 0%, k=1 -> 18%, k=2 -> 42%, k=3 -> 49%, k=4 -> 51%, k=5 -> 54%, k=6 -> 56%, k=7 -> 57%, k=8 -> 58%. The curve first surpasses the 50% VLA full-shot baseline at k=4 (reaching 51%)."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p032::chart::001",
            "page": 32,
            "modality": "chart",
            "figure_id": "Harness VLA Figure 6",
            "crop_files": ["data/advanced/crops/p032_img01_Im25.png"],
            "bbox": [52, 310, 558, 495],
            "text": (
                "[Harness VLA Figure 6 | Page 32 | Crop: p032_img01_Im25.png] "
                "Figure 6: Grouped bar chart 'Task completion attribution across benchmarks' (2559x842 px). "
                "Y-axis: 'Proportion of task completions (%)' from 0 to 100. Legend has 2 bar types: 'Analytic primitive finish' (blue) and 'VLA primitive finish' (orange).\n"
                "- Left panel (LIBERO): Spatial = 72% Analytic / 28% VLA; Object = 80% Analytic / 20% VLA; Goal = 79% Analytic / 21% VLA; LIBERO-10 = 84% Analytic / 16% VLA.\n"
                "- Middle panel (RoboCasa365): Composite seen = 40% Analytic / 60% VLA; Composite unseen = 17% Analytic / 83% VLA; Atomic = 14% Analytic / 86% VLA.\n"
                "- Right panel (RoboTwin): 50 tasks = 34% Analytic / 66% VLA."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p070::chart::001",
            "page": 70,
            "modality": "chart",
            "figure_id": "AMP Figure 4",
            "crop_files": [
                "data/advanced/crops/p070_img01_Im12.png",
                "data/advanced/crops/p070_img02_Im13.png",
            ],
            "bbox": [304, 210, 558, 430],
            "text": (
                "[AMP Fig. 4 | Page 70 | Crops: p070_img01_Im12.png, p070_img02_Im13.png] "
                "Fig. 4: Performance of Humanoid: Target Heading policies trained with three different datasets: 'Locomotion' (blue), 'Walk' (orange), and 'Run' (green).\n"
                "1) Left chart (p070_img01_Im12.png): X-axis 'Samples' (from 0.00 to 2.00 x 10^8 samples, i.e., 200 million samples); Y-axis 'Task Return' (0.0 to 1.0). 'Locomotion' (blue curve) converges highest around ~0.88–0.90, 'Run' (green curve) plateaus around ~0.63–0.67, and 'Walk' (orange curve) stays lowest around ~0.42–0.50.\n"
                "2) Right chart (p070_img02_Im13.png): Scatter plot of X-axis 'Target Speed (m/s)' (from 0 to 5 m/s) vs Y-axis 'Average Speed (m/s)' (from 0 to 5 m/s), with a gray diagonal y=x dashed line representing matching the target speed. 'Walk' (orange pluses +) stays flat around ~1.2–1.3 m/s across all target speeds 1–5 m/s; 'Run' (green triangles) cannot go slow and stays at ~2.8–3.0 m/s when Target Speed is 1–2.5 m/s before rising to ~4.2 m/s at Target Speed 5 m/s; 'Locomotion' (blue circles) closely tracks the diagonal dashed line across the entire 1 m/s to 4.5 m/s range."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p086::chart::001",
            "page": 86,
            "modality": "chart",
            "figure_id": "FBRL Figure 2",
            "crop_files": [
                "data/advanced/crops/p086_img01_Im3.png",
                "data/advanced/crops/p086_img02_Im4.png",
                "data/advanced/crops/p086_img03_Im5.png",
                "data/advanced/crops/p086_img04_Im6.png",
            ],
            "bbox": [48, 520, 558, 745],
            "text": (
                "[FBRL Figure 2 | Page 86 | Crops: p086_img01_Im3.png .. p086_img04_Im6.png] "
                "Figure 2: Learning curves for n x n Gridworld where n = 5, 10, 15, 20 with fixed horizons of 50, 100, 150, 200 steps respectively, averaged over 10 trials. "
                "X-axis: 'Training Steps' (0 to 60,000); Y-axis: 'Average Reward'. Legend compares 'DDQN' (solid magenta/purple line with circles) vs 'FBRL' (dashed blue line with circles). "
                "In n=5 (p086_img01_Im3.png, Y-axis -0.4 to 0.8), both FBRL and DDQN quickly converge to ~0.9 average reward within ~15,000–25,000 steps. "
                "As grid size increases to n=20 with horizon 200 (p086_img04_Im6.png, Y-axis -2.0 to 0.5), FBRL (blue dashed) rises from -2.0 to ~0.0 average reward by ~25,000–30,000 steps, whereas DDQN (magenta solid) stays stuck at -2.0 until ~10,000 steps and remains much lower around -1.5 to -1.0 even at 60,000 training steps."
            ),
        },
        {
            "id": f"{SOURCE_ID}::p087::chart::001",
            "page": 87,
            "modality": "chart",
            "figure_id": "FBRL Figure 3",
            "crop_files": [
                "data/advanced/crops/p087_img01_Im7.png",
                "data/advanced/crops/p087_img02_Im8.png",
            ],
            "bbox": [90, 520, 515, 745],
            "text": (
                "[FBRL Figure 3 | Page 87 | Crops: p087_img01_Im7.png, p087_img02_Im8.png] "
                "Figure 3: Learning curves for Towers of Hanoi where n = 2 and n = 3 discs (fixed horizons of 50 and 100 steps respectively, averaged over 10 trials). "
                "X-axis: 'Training Steps' (0 to 40,000); Y-axis: 'Average Reward' (-0.4 to 1.0 for n=2; -2.0 to 1.0 for n=3). Legend compares 'DDQN' (solid magenta) vs 'FBRL' (dashed blue). "
                "For n=2 discs (p087_img01_Im7.png), both FBRL and DDQN reach the optimal ~0.95–0.97 average reward by ~8,000–15,000 steps, with FBRL rising slightly earlier between 2,000–8,000 steps. "
                "For n=3 discs (p087_img02_Im8.png), FBRL (dashed blue) rises much faster than DDQN between 3,000 and 15,000 training steps (reaching ~0.93 by 15,000 steps while DDQN is still at ~-0.8), though after ~27,000 steps FBRL experiences a drop/degradation (fluctuating around -0.5 to 0.5) which the text attributes to possible overfitting, while DDQN reaches ~0.93 after 27,000 steps."
            ),
        },
    ]


def build_chunks_and_manifest(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    page_hashes: dict[str, int] = {}
    duplicate_pages: list[dict[str, Any]] = []
    reference_only_pages: list[int] = [11, 12, 13, 14, 38, 39, 40, 41, 42, 43, 74, 75, 76, 88]
    dense_plot_grid_pages: list[int] = [19, 80, 81]

    chunks: list[dict[str, Any]] = []

    for idx in range(num_pages):
        page_num = idx + 1
        raw_text = (reader.pages[idx].extract_text() or "").strip()
        norm_hash = hashlib.sha256(re.sub(r"\s+", " ", raw_text).encode("utf-8")).hexdigest()
        if norm_hash in page_hashes:
            duplicate_pages.append(
                {
                    "page": page_num,
                    "duplicate_of_page": page_hashes[norm_hash],
                    "reason": "Exact duplicate attachment of Paper 3 (AMP: Adversarial Motion Priors, pp. 62-81).",
                }
            )
            continue
        page_hashes[norm_hash] = page_num

        # Split page text into ~1600-char overlapping semantic segments without dropping any text
        paper = _paper_for_page(page_num)
        dehyphenated = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", raw_text)
        clean_page_text = re.sub(r"[ \t]+", " ", dehyphenated)
        lines = [ln.strip() for ln in clean_page_text.splitlines() if ln.strip()]
        buf_lines: list[str] = []
        buf_len = 0
        seg_idx = 0
        for ln in lines:
            if buf_len + len(ln) + 1 <= 1650 or not buf_lines:
                buf_lines.append(ln)
                buf_len += len(ln) + 1
            else:
                seg_text = " ".join(buf_lines)
                chunks.append(
                    {
                        "id": f"{SOURCE_ID}::p{page_num:03d}::text::{seg_idx:03d}",
                        "source_id": SOURCE_ID,
                        "source": SOURCE_FILE,
                        "page": page_num,
                        "text": f"[{paper} | Page {page_num}] {seg_text}",
                        "metadata": {
                            "paper": paper,
                            "modality": "text",
                            "is_reference_page": page_num in reference_only_pages,
                        },
                    }
                )
                seg_idx += 1
                # Keep last 8 lines as overlap so sentences crossing segment boundaries remain intact
                buf_lines = buf_lines[-8:] + [ln]
                buf_len = sum(len(x) + 1 for x in buf_lines)
        if buf_lines:
            seg_text = " ".join(buf_lines)
            chunks.append(
                {
                    "id": f"{SOURCE_ID}::p{page_num:03d}::text::{seg_idx:03d}",
                    "source_id": SOURCE_ID,
                    "source": SOURCE_FILE,
                    "page": page_num,
                    "text": f"[{paper} | Page {page_num}] {seg_text}",
                    "metadata": {
                        "paper": paper,
                        "modality": "text",
                        "is_reference_page": page_num in reference_only_pages,
                    },
                }
            )

    # Add structured tables
    for tbl in _structured_tables():
        p = tbl["page"]
        chunks.append(
            {
                "id": tbl["id"],
                "source_id": SOURCE_ID,
                "source": SOURCE_FILE,
                "page": p,
                "text": tbl["text"],
                "metadata": {
                    "paper": _paper_for_page(p),
                    "modality": "table",
                    "table_id": tbl["table_id"],
                    "bbox": tbl["bbox"],
                    "units": tbl["units"],
                },
            }
        )

    # Add visual figure/diagram and chart chunks
    for vis in _visual_items():
        p = vis["page"]
        chunks.append(
            {
                "id": vis["id"],
                "source_id": SOURCE_ID,
                "source": SOURCE_FILE,
                "page": p,
                "text": vis["text"],
                "metadata": {
                    "paper": _paper_for_page(p),
                    "modality": vis["modality"],
                    "figure_id": vis["figure_id"],
                    "crop_files": vis["crop_files"],
                    "bbox": vis["bbox"],
                },
            }
        )

    manifest = {
        "document": {
            "source_id": SOURCE_ID,
            "file_name": SOURCE_FILE,
            "total_pages": num_pages,
            "unique_indexed_pages": 88,
            "duplicate_pages_skipped": len(duplicate_pages),
            "license": "CC-BY 4.0 (arXiv / ACM Open Access)",
            "papers_included": [
                {
                    "paper_id": "P1_AWR",
                    "title": "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning",
                    "authors": "Xue Bin Peng, Aviral Kumar, Grace Zhang, Sergey Levine",
                    "pages": "1-19",
                    "source_url": "https://arxiv.org/abs/1910.00177",
                },
                {
                    "paper_id": "P2_HarnessVLA",
                    "title": "Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents",
                    "authors": "Yixian Zhang, Huanming Zhang, et al.",
                    "pages": "20-61",
                    "source_url": "https://arxiv.org/abs/2607.08448v5",
                },
                {
                    "paper_id": "P3_AMP",
                    "title": "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control",
                    "authors": "Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa",
                    "pages": "62-81",
                    "source_url": "https://doi.org/10.1145/3450626.3459670",
                },
                {
                    "paper_id": "P4_FBRL",
                    "title": "Forward-Backward Reinforcement Learning",
                    "authors": "Ashley D. Edwards, Laura Downs, James C. Davidson",
                    "pages": "82-88",
                    "source_url": "https://arxiv.org/abs/1803.10227v1",
                },
            ],
        },
        "index_statistics": {
            "total_chunks": len(chunks),
            "by_modality": {
                "text": sum(1 for c in chunks if c["metadata"]["modality"] == "text"),
                "table": sum(1 for c in chunks if c["metadata"]["modality"] == "table"),
                "figure_diagram": sum(1 for c in chunks if c["metadata"]["modality"] == "figure_diagram"),
                "chart": sum(1 for c in chunks if c["metadata"]["modality"] == "chart"),
            },
            "extracted_image_crops": 174,
            "crop_directory": "data/advanced/crops/",
        },
        "unprocessed_or_special_pages": {
            "duplicate_pages_89_to_108": {
                "pages": "89-108",
                "reason": "Pages 89-108 are an exact byte/text duplicate of Paper 3 (AMP, pages 62-81) concatenated at the end of RL.pdf.",
                "handling_and_impact": "Detected via SHA-256 normalized text hashing during ingestion and deduplicated so identical chunks from pp. 89-108 do not pollute top-k retrieval rankings or waste context budget.",
                "details": duplicate_pages,
            },
            "bibliography_pages": {
                "pages": reference_only_pages,
                "reason": "Pages containing bibliographic reference lists rather than experimental findings.",
                "handling_and_impact": "Indexed with metadata flag is_reference_page=True so citation-Lookup remains possible while main experimental queries prioritize content/table/visual chunks.",
            },
            "appendix_thumbnail_grids": {
                "pages": dense_plot_grid_pages,
                "reason": "Pages 19, 80, and 81 contain 12-21 small sub-chart thumbnails with minimal body text (<620 chars).",
                "handling_and_impact": "All 45 sub-chart images on pages 19, 80, and 81 were extracted to data/advanced/crops/ and their captions indexed, while primary quantitative comparisons use the full-resolution main-text charts and tables.",
            },
        },
    }
    return chunks, manifest


def build_benchmark() -> list[dict[str, Any]]:
    """Build 19 dev cases (16 answerable + 3 unanswerable) and 30 test cases (24 answerable + 6 unanswerable)."""
    cases: list[dict[str, Any]] = [
        # ==================== DEV SPLIT (19 cases: 16 answerable + 3 unanswerable) ====================
        # Dev Table (4)
        {
            "id": "adv_dev_tab01",
            "split": "dev",
            "primary_group": "table",
            "secondary_tags": ["AWR", "OpenAI Gym"],
            "answerable": True,
            "question": "In AWR Table 1 (page 8), what is the final average return and standard deviation of AWR (Ours) vs SAC on the Hopper-v2 task, and how many points higher is AWR's mean return than SAC's?",
            "expected_answer": "On Hopper-v2 (AWR Table 1, page 8), AWR (Ours) achieves 3405 ± 121 and SAC achieves 2769 ± 552. AWR's mean return is 636 points higher than SAC's (3405 - 2769 = 636).",
            "source": SOURCE_FILE,
            "pages": [8],
            "evidence_details": "Page 8, AWR Table 1, row 'Hopper-v2', columns 'SAC' (2769 ± 552) and 'AWR (Ours)' (3405 ± 121); unit: average episode return ± 1 SD across 5 seeds; calculation: 3405 - 2769 = 636.",
            "required_facts": ["3405", "2769", "636"],
            "gold": [{"source_id": SOURCE_ID, "page": 8, "evidence": "| Hopper-v2 | 1880 ± 337 | 1391 ± 304 | 855 ± 282 | 935 ± 489 | 2769 ± 552 | 605 ± 114 | 3405 ± 121 |"}],
        },
        {
            "id": "adv_dev_tab02",
            "split": "dev",
            "primary_group": "table",
            "secondary_tags": ["Harness VLA", "Standard LIBERO"],
            "answerable": True,
            "question": "According to Harness VLA Table 2 (page 27), what is the success rate (%) of Harness VLA (CC) on the Object suite and Overall on standard LIBERO, and how does its Overall score compare to AtomVLA?",
            "expected_answer": "In Harness VLA Table 2 (page 27), Harness VLA (CC) achieves 100.0% on the Object suite and 96.0% Overall on standard LIBERO, which is 1.0 percentage point lower than AtomVLA's 97.0% Overall (97.0% - 96.0% = 1.0%).",
            "source": SOURCE_FILE,
            "pages": [27],
            "evidence_details": "Page 27, Harness VLA Table 2, rows 'AtomVLA' (Object 99.6%, Overall 97.0%) and 'Harness VLA (CC)' (Object 100.0%, Overall 96.0%); unit: success rate (%) over 100 trials/suite.",
            "required_facts": ["100.0", "96.0", "97.0"],
            "gold": [{"source_id": SOURCE_ID, "page": 27, "evidence": "| Harness VLA (CC) | 97.0 | 100.0 | 94.0 | 93.0 | 96.0 |"}],
        },
        {
            "id": "adv_dev_tab03",
            "split": "dev",
            "primary_group": "table",
            "secondary_tags": ["AMP", "Motion Dataset"],
            "answerable": True,
            "question": "In AMP Table 2 (page 70), what are the dataset size (in seconds), number of clips, and number of subjects for the Humanoid 'Walk + Punch' dataset?",
            "expected_answer": "According to AMP Table 2 (page 70), the Humanoid 'Walk + Punch' dataset has a total size of 247.8 seconds (s), consisting of 15 clips recorded from 9 subjects.",
            "source": SOURCE_FILE,
            "pages": [70],
            "evidence_details": "Page 70, AMP Table 2, row 'Humanoid | Walk + Punch': Size (s) = 247.8, Clips = 15, Subjects = 9.",
            "required_facts": ["247.8", "15", "9"],
            "gold": [{"source_id": SOURCE_ID, "page": 70, "evidence": "| Humanoid | Walk + Punch | 247.8 | 15 | 9 |"}],
        },
        {
            "id": "adv_dev_tab04",
            "split": "dev",
            "primary_group": "table",
            "secondary_tags": ["AMP", "Spatial Composition"],
            "answerable": True,
            "question": "In AMP Table 6 (page 79), what are the Heading Return and Waving Return for the policy trained on the combined 'Wave + Walk (281.4s)' dataset compared to training on 'Walk (229.7s)' alone?",
            "expected_answer": "In AMP Table 6 (page 79), the policy trained on 'Wave + Walk (281.4s)' achieves a Heading Return of 0.885 ± 0.184 and a Waving Return of 0.891 ± 0.202, whereas training on 'Walk (229.7s)' alone achieves a Heading Return of 0.945 ± 0.192 but drops to a Waving Return of 0.306 ± 0.378.",
            "source": SOURCE_FILE,
            "pages": [79],
            "evidence_details": "Page 79, AMP Table 6, rows 'Walk (229.7s)' (0.945 ± 0.192, 0.306 ± 0.378) and 'Wave + Walk (281.4s)' (0.885 ± 0.184, 0.891 ± 0.202); unit: normalized task return averaged across 100 episodes.",
            "required_facts": ["0.885", "0.891", "0.945", "0.306"],
            "gold": [{"source_id": SOURCE_ID, "page": 79, "evidence": "| Wave + Walk (281.4s) | 0.885 ± 0.184 | 0.891 ± 0.202 |"}],
        },
        # Dev Figure/Diagram (3)
        {
            "id": "adv_dev_fig01",
            "split": "dev",
            "primary_group": "figure_diagram",
            "secondary_tags": ["AWR", "Character Snapshots"],
            "answerable": True,
            "question": "In AWR Figure 5 (page 9), what are the degrees of freedom (DoF) of the simulated humanoid and dog characters, and which four motion imitation skills are shown in subfigures (a)–(d)?",
            "expected_answer": "AWR Figure 5 (page 9) shows a 34 DoF humanoid and an 82 DoF dog performing four skills: (a) Humanoid: Cartwheel, (b) Humanoid: Spinkick, (c) Dog: Trot, and (d) Dog: Turn.",
            "source": SOURCE_FILE,
            "pages": [9],
            "evidence_details": "Page 9, AWR Figure 5 caption and crops p009_img05_Im15.png..p009_img08_Im18.png: 34 DoF humanoid and 82 DoF dog; subfigures (a) Humanoid: Cartwheel, (b) Humanoid: Spinkick, (c) Dog: Trot, (d) Dog: Turn.",
            "required_facts": ["34", "82", "Cartwheel", "Spinkick", "Trot", "Turn"],
            "gold": [{"source_id": SOURCE_ID, "page": 9, "evidence": "(a) Humanoid: Cartwheel, (b) Humanoid: Spinkick, (c) Dog: Trot, and (d) Dog: Turn."}],
        },
        {
            "id": "adv_dev_fig02",
            "split": "dev",
            "primary_group": "figure_diagram",
            "secondary_tags": ["Harness VLA", "Trajectory Distribution Diagram"],
            "answerable": True,
            "question": "In Harness VLA Figure 2 (page 22), what is the exact task description shown in the left box, and what do the green circles and orange circles represent in the rollout diagram?",
            "expected_answer": "In Harness VLA Figure 2 (page 22, crop p022_img01_Im3.png), the task description is 'Pick the pan and the sponge and place them into the sink. Then turn on the water.' In the rollout legend, green circles represent 'Analytic Primitive Control' (which perceives targets and moves the robot between regions) and orange circles represent 'VLA Contact-Rich Control' (inside the orange-hatched 'VLA in Distribution Trajectories' regions).",
            "source": SOURCE_FILE,
            "pages": [22],
            "evidence_details": "Page 22, Figure 2 diagram crop p022_img01_Im3.png: left box text and bottom-right legend ('Analytic Primitive Control' green circle, 'VLA Contact-Rich Control' orange circle).",
            "required_facts": ["pan", "sponge", "sink", "Analytic Primitive Control", "VLA Contact-Rich Control"],
            "gold": [{"source_id": SOURCE_ID, "page": 22, "evidence": "Pick the pan and the sponge and place them into the sink. Then turn on the water."}],
        },
        {
            "id": "adv_dev_fig03",
            "split": "dev",
            "primary_group": "figure_diagram",
            "secondary_tags": ["AMP", "Non-humanoid Characters"],
            "answerable": True,
            "question": "In AMP Fig. 7 (page 72), what are the two non-humanoid characters and their exact degrees of freedom (DoF), and what three locomotion skills are depicted in subfigures (a), (b), and (c)?",
            "expected_answer": "AMP Fig. 7 (page 72, crops p072_img01_Im25.png..p072_img03_Im27.png) depicts a 59 DoF T-Rex and a 64 DoF dog performing: (a) T-Rex (Walk), (b) Dog (Trot), and (c) Dog (Canter).",
            "source": SOURCE_FILE,
            "pages": [72],
            "evidence_details": "Page 72, Fig. 7 caption and crops p072_img01_Im25.png..p072_img03_Im27.png: 59 DoF T-Rex and 64 DoF dog; (a) T-Rex (Walk), (b) Dog (Trot), (c) Dog (Canter).",
            "required_facts": ["59", "64", "T-Rex", "Trot", "Canter"],
            "gold": [{"source_id": SOURCE_ID, "page": 72, "evidence": "59 DoF (degrees of freedom) T-Rex and a 64 DoF dog"}],
        },
        # Dev Chart (3)
        {
            "id": "adv_dev_cht01",
            "split": "dev",
            "primary_group": "chart",
            "secondary_tags": ["AWR", "Ablation Chart"],
            "answerable": True,
            "question": "In the left chart of AWR Figure 4 ('Ablation: HalfCheetah-v2', page 9), what are the X-axis and Y-axis labels, and what is the ranking of the 5 curves from highest to lowest return at 2.0 x 10^7 samples?",
            "expected_answer": "In AWR Figure 4 left ('Ablation: HalfCheetah-v2', page 9, crop p009_img01_Im11.png), the X-axis is 'Samples' (up to 2.0 x 10^7) and the Y-axis is 'Return'. From highest to lowest return at 2.0 x 10^7 samples, the 5 curves rank: 1) AWR (Ours) (~9,100), 2) No TD(lambda) (~5,900), 3) On-Policy (~4,500), 4) No Baseline (~3,600), and 5) RWR (~1,400).",
            "source": SOURCE_FILE,
            "pages": [9],
            "evidence_details": "Page 9, Figure 4 left crop p009_img01_Im11.png: X-axis 'Samples' (1e7), Y-axis 'Return', legend & curve order: AWR (Ours) > No TD(lambda) > On-Policy > No Baseline > RWR.",
            "required_facts": ["Samples", "Return", "AWR", "No TD", "On-Policy", "No Baseline", "RWR"],
            "gold": [{"source_id": SOURCE_ID, "page": 9, "evidence": "'AWR (Ours)' (red) achieves highest return (~9,100 at 2.0x10^7 samples), followed by 'No TD(lambda)' (green, ~5,900), 'On-Policy' (blue, ~4,500), 'No Baseline' (orange, ~3,600), and 'RWR' (purple, lowest at ~1,400)"}],
        },
        {
            "id": "adv_dev_cht02",
            "split": "dev",
            "primary_group": "chart",
            "secondary_tags": ["Harness VLA", "Figure 4c RoboTwin"],
            "answerable": True,
            "question": "In Harness VLA Figure 4 panel (c) (ROBOTWIN C2R, page 30), what are the cumulative task success rates (%) at k = 1, 2, 3, and 4 VLA primitive invocations, and at what k does Harness VLA first surpass the 'VLA full-shot (50%)' baseline?",
            "expected_answer": "In Harness VLA Figure 4(c) (ROBOTWIN C2R, page 30, crop p030_img03_Im12.png), the cumulative task success rates are 18% at k=1, 42% at k=2, 49% at k=3, and 51% at k=4. Harness VLA first surpasses the 'VLA full-shot (50%)' baseline at k = 4 (reaching 51%).",
            "source": SOURCE_FILE,
            "pages": [30],
            "evidence_details": "Page 30, Figure 4(c) crop p030_img03_Im12.png: X-axis 'VLA primitive invocations (<= k)', Y-axis 'Cumulative task success (%)', values: k=1 -> 18%, k=2 -> 42%, k=3 -> 49%, k=4 -> 51% > 50% blue dashed line.",
            "required_facts": ["18", "42", "49", "51", "4"],
            "gold": [{"source_id": SOURCE_ID, "page": 30, "evidence": "k=0 -> 0%, k=1 -> 18%, k=2 -> 42%, k=3 -> 49%, k=4 -> 51%, k=5 -> 54%, k=6 -> 56%, k=7 -> 57%, k=8 -> 58%"}],
        },
        {
            "id": "adv_dev_cht03",
            "split": "dev",
            "primary_group": "chart",
            "secondary_tags": ["FBRL", "Towers of Hanoi Chart"],
            "answerable": True,
            "question": "In FBRL Figure 3 (page 87), what are the fixed step horizons for n = 2 and n = 3 discs in Towers of Hanoi, and what happens to the FBRL curve after ~27,000 training steps for n = 3 discs?",
            "expected_answer": "In FBRL Figure 3 (page 87, crops p087_img01_Im7.png and p087_img02_Im8.png), the fixed horizons are 50 steps for n = 2 discs and 100 steps for n = 3 discs (averaged over 10 trials). For n = 3 discs, FBRL learns much faster than DDQN early on (reaching ~0.93 by 15,000 steps), but after ~27,000 training steps FBRL's average reward degrades/drops (fluctuating around -0.5 to 0.5), which the authors attribute to possible overfitting.",
            "source": SOURCE_FILE,
            "pages": [86, 87],
            "evidence_details": "Pages 86-87, Figure 3 caption, Section 5.2 text, and crop p087_img02_Im8.png: horizons 50 and 100 steps; FBRL degrades for 3 discs after ~27,000 steps due to overfitting.",
            "required_facts": ["50", "100", "overfitting"],
            "gold": [{"source_id": SOURCE_ID, "page": 87, "evidence": "For n=3 discs (p087_img02_Im8.png), FBRL (dashed blue) rises much faster than DDQN between 3,000 and 15,000 training steps"}],
        },
        # Dev Multi-hop (3)
        {
            "id": "adv_dev_mhp01",
            "split": "dev",
            "primary_group": "multi_hop",
            "secondary_tags": ["AWR", "AMP", "Cross-Paper Comparison"],
            "answerable": True,
            "question": "Compare the simulated dog character used in the AWR paper (page 9, Section 5.3 / Figure 5) with the simulated dog character used in the AMP paper (page 72, Fig. 7): how many degrees of freedom (DoF) does the dog have in each paper, and what is the difference in DoF?",
            "expected_answer": "In the AWR paper (page 9, Figure 5), the simulated dog has 82 DoF, whereas in the AMP paper (page 72, Fig. 7), the simulated dog has 64 DoF. The difference is 18 degrees of freedom (82 - 64 = 18 DoF).",
            "source": SOURCE_FILE,
            "pages": [9, 72],
            "evidence_details": "Multi-hop combining Page 9 (AWR Figure 5: '82 DoF dog') and Page 72 (AMP Fig. 7: '64 DoF dog'); calculation: 82 - 64 = 18 DoF.",
            "required_facts": ["82", "64", "18"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 9, "evidence": "82 DoF dog"},
                {"source_id": SOURCE_ID, "page": 72, "evidence": "64 DoF dog"},
            ],
        },
        {
            "id": "adv_dev_mhp02",
            "split": "dev",
            "primary_group": "multi_hop",
            "secondary_tags": ["Harness VLA", "Zero-shot vs Few-shot Memory"],
            "answerable": True,
            "question": "Combining Harness VLA Table 3 (page 27) and Table 5 (page 29), how much does Harness VLA (CC) success rate drop on LIBERO-Pro GOAL under instruction-redirection (Goal-T) versus position-swap (Goal-S) when Task Specific Memory is removed in the zero-shot setting?",
            "expected_answer": "From Table 3 (page 27) to Table 5 (page 29), removing Task Specific Memory (zero-shot Harness VLA CC) reduces success on Goal-T (instruction-redirection) by only 8.0 percentage points (from 87.0% few-shot to 79.0% zero-shot), whereas on Goal-S (position-swap) it drops by 56.0 percentage points (from 87.0% few-shot to 31.0% zero-shot).",
            "source": SOURCE_FILE,
            "pages": [27, 28, 29],
            "evidence_details": "Multi-hop combining Page 27 Table 3 (Harness VLA CC Goal-T = 87.0%, Goal-S = 87.0%) and Page 28-29 Table 5 (Zero-shot Harness VLA CC Task-T = 79.0%, Pos-S = 31.0%); calculations: 87.0 - 79.0 = 8.0% and 87.0 - 31.0 = 56.0%.",
            "required_facts": ["87.0", "79.0", "31.0", "56.0"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 27, "evidence": "| Harness VLA (CC) | 94.0 | 80.0 | 88.0 | 90.0 | 87.0 | 87.0 | 71.0 | 62.0 | 82.4 |"},
                {"source_id": SOURCE_ID, "page": 29, "evidence": "Pos (S) Average: Cap-X = 25.6%, Zero-shot Harness VLA (CC) = 31.0%"},
            ],
        },
        {
            "id": "adv_dev_mhp03",
            "split": "dev",
            "primary_group": "multi_hop",
            "secondary_tags": ["AMP", "Target Heading vs Dataset Size"],
            "answerable": True,
            "question": "Combining AMP Table 1 and Table 2 (page 70), for the Humanoid Target Heading task, how many times larger in duration (seconds) is the 'Locomotion' dataset than the 'Zombie' dataset, and which of the two achieves a higher normalized Task Return?",
            "expected_answer": "On page 70, AMP Table 2 shows the Humanoid 'Locomotion' dataset is 434.1s (56 clips, 8 subjects) while the 'Zombie' dataset is 18.3s (1 clip, 1 subject), making Locomotion ~23.72 times larger in duration (434.1 / 18.3 = 23.72). In AMP Table 1, 'Zombie' achieves a slightly higher normalized Task Return on Target Heading (0.94 ± 0.00) than 'Locomotion' (0.90 ± 0.01).",
            "source": SOURCE_FILE,
            "pages": [70],
            "evidence_details": "Multi-hop across AMP Table 1 (Target Heading: Locomotion = 0.90 ± 0.01, Zombie = 0.94 ± 0.00) and AMP Table 2 (Locomotion = 434.1s, Zombie = 18.3s; ratio = 434.1 / 18.3 = 23.72).",
            "required_facts": ["434.1", "18.3", "0.94", "0.90"],
            "gold": [{"source_id": SOURCE_ID, "page": 70, "evidence": "| Humanoid | Locomotion | 434.1 | 56 | 8 |"}],
        },
        # Dev Text (3)
        {
            "id": "adv_dev_txt01",
            "split": "dev",
            "primary_group": "text",
            "secondary_tags": ["AWR", "Algorithm 1 & Objective"],
            "answerable": True,
            "question": "According to Section 3.1 of the AWR paper (page 3), how does the optimization objective of AWR differ from that of RWR and REPS, and what exponentiated weight is assigned to each state-action pair in Algorithm 1?",
            "expected_answer": "According to page 3 (Section 3.1 and Algorithm 1), RWR and REPS maximize the expected return J(pi), whereas AWR maximizes the expected improvement eta(pi) = J(pi) - J(mu) over a sampling policy mu(a|s). In Algorithm 1, each state-action pair in the replay buffer D is weighted by the exponentiated advantage exp((1 / beta) * (R_{s,a}^D - V_k^D(s))) = exp((1 / beta) * A^D(s, a)).",
            "source": SOURCE_FILE,
            "pages": [3],
            "evidence_details": "Page 3, Algorithm 1 line 6 and Section 3.1 paragraph 1: expected improvement eta(pi) = J(pi) - J(mu) vs expected return J(pi); weight exp((1/beta)*(R_{s,a}^D - V(s))).",
            "required_facts": ["expected improvement", "expected return", "exp", "beta"],
            "gold": [{"source_id": SOURCE_ID, "page": 3, "evidence": "which maximize the expected return J(π) instead of the expected improvement."}],
        },
        {
            "id": "adv_dev_txt02",
            "split": "dev",
            "primary_group": "text",
            "secondary_tags": ["Harness VLA", "Flash Mode Spatial Binding"],
            "answerable": True,
            "question": "According to Appendix G of Harness VLA (page 60), under what horizontal distance condition is a MOVE_TO or MOVE_POSE waypoint stored relative to an anchor in Flash Mode, and what is the maximum number of grasp attempts allowed during execution?",
            "expected_answer": "According to Appendix G (page 60, 'Spatial binding and replay'), in the LIBERO backend of Flash Mode, a MOVE_TO or MOVE_POSE waypoint within 0.20 m of an anchor in the horizontal plane is stored relative to that anchor (Delta x_i = x_i^ref - a_{j(i)}^ref). During execution, Flash Mode allows at most 3 (three) grasp attempts, skips dependent transport after an unconfirmed grasp, and stops if a required anchor cannot be localized.",
            "source": SOURCE_FILE,
            "pages": [60],
            "evidence_details": "Page 60, Appendix G 'Spatial binding and replay': waypoint within 0.20m of an anchor in horizontal plane; allows at most three grasp attempts.",
            "required_facts": ["0.20", "three"],
            "gold": [{"source_id": SOURCE_ID, "page": 60, "evidence": "Execution allows at most three grasp attempts, skips dependent transport after an unconfirmed grasp"}],
        },
        {
            "id": "adv_dev_txt03",
            "split": "dev",
            "primary_group": "text",
            "secondary_tags": ["AMP", "Ablation & Failure Modes"],
            "answerable": True,
            "question": "According to Section 8.4 and Section 8.5 of the AMP paper (pages 72–73), why does AMP converge to shuffling forward instead of flipping on the 'Front-Flip' motion, and what happens on the 'Roll' motion if velocity features are removed from the discriminator?",
            "expected_answer": "According to pages 72–73 (Sections 8.4 and 8.5), on 'Front-Flip' AMP is prone to converging to a locally optimal behavior of shuffling forward to avoid falling because AMP is not synchronized with a reference motion via a phase variable and therefore cannot use early episode termination based on pose deviation. On 'Roll', if velocity features are removed from the discriminator's observations, the character converges to holding a fixed pose on the ground instead of performing a roll.",
            "source": SOURCE_FILE,
            "pages": [72, 73],
            "evidence_details": "Page 72 Section 8.4 (Front-Flip local optimum: shuffling forward, no phase synchronization for early termination) and Page 73 Section 8.5 (absence of velocity features on roll leads to holding a fixed pose on the ground).",
            "required_facts": ["shuffle", "phase", "fixed pose"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 72, "evidence": "instead of performing a flip, the character learns to simply shuffle forwards in order to avoid falling."},
                {"source_id": SOURCE_ID, "page": 73, "evidence": "in the absence of velocity features, the character is prone to converging to a strategy of holding a fixed pose on the ground, instead of performing a roll."},
            ],
        },
        # Dev Unanswerable (3)
        {
            "id": "adv_dev_una01",
            "split": "dev",
            "primary_group": "unanswerable",
            "secondary_tags": ["AWR", "Atari / ALE"],
            "answerable": False,
            "question": "In the AWR paper, what is the human-normalized median score achieved by AWR across the 57 Atari 2600 Arcade Learning Environment (ALE) games?",
            "expected_answer": "Abstain / Insufficient evidence: The AWR paper in RL.pdf evaluates OpenAI Gym control tasks (Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2, LunarLander-v2, Walker2d-v2), motion imitation (Humanoid, Dog), and static off-policy Gym datasets, and does not report any experiments or scores on Atari 2600 / ALE games.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by searching all 108 pages of RL.pdf for 'Atari', 'ALE', and 'Arcade': zero occurrences in AWR (pages 1-19).",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_dev_una02",
            "split": "dev",
            "primary_group": "unanswerable",
            "secondary_tags": ["Harness VLA", "Token API Cost"],
            "answerable": False,
            "question": "In Harness VLA Table 7, what is the exact dollar API cost ($ USD) per episode for running Codex (reasoning high) versus Claude Code (Opus-4.7) on LIBERO-Pro?",
            "expected_answer": "Abstain / Insufficient evidence: Harness VLA Table 7 (page 35) and Appendix G (page 61) report success rates (%) and execution time in seconds (529.23s vs 344.17s vs 56.17s), but RL.pdf never reports dollar API costs ($ USD) or token billing costs per episode.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by inspecting Table 7 (page 35), Appendix G (pages 60-61), and searching RL.pdf for '$', 'USD', 'cost', 'billing', 'price': no dollar API cost is reported.",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_dev_una03",
            "split": "dev",
            "primary_group": "unanswerable",
            "secondary_tags": ["FBRL", "MuJoCo"],
            "answerable": False,
            "question": "In the Forward-Backward Reinforcement Learning (FBRL) paper (pages 82–88), what average return does FBRL achieve on the MuJoCo FetchPickAndPlace-v1 robotic arm benchmark?",
            "expected_answer": "Abstain / Insufficient evidence: The FBRL paper (pages 82–88) only evaluates two discrete environments—n x n Gridworld (n=5, 10, 15, 20) and n-disc Towers of Hanoi (n=2, 3)—and does not evaluate FetchPickAndPlace-v1 or any MuJoCo robotic arm benchmark.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by inspecting pages 82-88 (FBRL Section 5 Experiments): only Gridworld (Section 5.1) and Towers of Hanoi (Section 5.2) are evaluated.",
            "required_facts": [],
            "gold": [],
        },

        # ==================== TEST SPLIT (30 cases: 24 answerable + 6 unanswerable) ====================
        # --- Group 1: TABLE (6 cases: adv_test_tab01 .. adv_test_tab06) ---
        {
            "id": "adv_test_tab01",
            "split": "test",
            "primary_group": "table",
            "secondary_tags": ["AWR", "Table 1", "OpenAI Gym"],
            "answerable": True,
            "question": "According to AWR Table 1 (page 8), on the Walker2d-v2 task, what are the final average returns (± SD) of AWR (Ours), SAC, and RWR, and by how many points does AWR (Ours) outperform RWR?",
            "expected_answer": "In AWR Table 1 (page 8) on Walker2d-v2, AWR (Ours) achieves 5813 ± 483, SAC achieves 5805 ± 587, and RWR achieves 406 ± 64 (in units of average episode return ± 1 SD across 5 random seeds). AWR outperforms RWR by 5407 return points (5813 - 406 = 5407).",
            "source": SOURCE_FILE,
            "pages": [8],
            "evidence_details": "Page 8, AWR Table 1, row 'Walker2d-v2': SAC = 5805 ± 587, RWR = 406 ± 64, AWR (Ours) = 5813 ± 483; calculation: 5813 - 406 = 5407.",
            "required_facts": ["5813", "5805", "406", "5407"],
            "gold": [{"source_id": SOURCE_ID, "page": 8, "evidence": "| Walker2d-v2 | 2765 ± 168 | 2617 ± 362 | 401 ± 470 | 4212 ± 427 | 5805 ± 587 | 406 ± 64 | 5813 ± 483 |"}],
        },
        {
            "id": "adv_test_tab02",
            "split": "test",
            "primary_group": "table",
            "secondary_tags": ["AWR", "Table 2", "Motion Imitation"],
            "answerable": True,
            "question": "In AWR Table 2 (page 10), what are the normalized returns (± SD) of PPO, RWR, and AWR (Ours) on 'Humanoid: Spinkick' and 'Dog: Canter', and what is the difference in mean normalized return between AWR and RWR on 'Humanoid: Spinkick'?",
            "expected_answer": "In AWR Table 2 (page 10), on 'Humanoid: Spinkick', PPO achieves 0.70 ± 0.02, RWR achieves 0.05 ± 0.03, and AWR (Ours) achieves 0.77 ± 0.04 (a difference of 0.77 - 0.05 = 0.72 higher for AWR than RWR). On 'Dog: Canter', PPO achieves 0.76 ± 0.03, RWR achieves 0.78 ± 0.04, and AWR (Ours) achieves 0.86 ± 0.01.",
            "source": SOURCE_FILE,
            "pages": [10],
            "evidence_details": "Page 10, AWR Table 2, rows 'Humanoid: Spinkick' (PPO 0.70±0.02, RWR 0.05±0.03, AWR 0.77±0.04; diff = 0.72) and 'Dog: Canter' (PPO 0.76±0.03, RWR 0.78±0.04, AWR 0.86±0.01).",
            "required_facts": ["0.77", "0.05", "0.72", "0.86"],
            "gold": [{"source_id": SOURCE_ID, "page": 10, "evidence": "| Humanoid: Spinkick | 0.70 ± 0.02 | 0.05 ± 0.03 | 0.77 ± 0.04 |"}],
        },
        {
            "id": "adv_test_tab03",
            "split": "test",
            "primary_group": "table",
            "secondary_tags": ["Harness VLA", "Table 3", "LIBERO-Pro"],
            "answerable": True,
            "question": "According to Harness VLA Table 3 (page 27), what are the Overall LIBERO-Pro success rates (%) of RATS, pi_RLinf, Harness VLA (Codex), and Harness VLA (CC), and by how many percentage points does Harness VLA (CC) improve over RATS and over pi_RLinf?",
            "expected_answer": "In Harness VLA Table 3 (page 27), the Overall LIBERO-Pro success rates are: RATS = 43.8%, pi_RLinf = 50.0%, Harness VLA (Codex) = 72.1%, and Harness VLA (CC) = 82.4%. Harness VLA (CC) improves over RATS by 38.6 percentage points (82.4% - 43.8% = 38.6%) and over pi_RLinf by 32.4 percentage points (82.4% - 50.0% = 32.4%).",
            "source": SOURCE_FILE,
            "pages": [27, 28],
            "evidence_details": "Page 27, Harness VLA Table 3, column 'Overall': RATS = 43.8%, pi_RLinf = 50.0%, Harness VLA (Codex) = 72.1%, Harness VLA (CC) = 82.4%; differences: 82.4 - 43.8 = 38.6 percentage points, 82.4 - 50.0 = 32.4 percentage points.",
            "required_facts": ["43.8", "50.0", "72.1", "82.4", "38.6", "32.4"],
            "gold": [{"source_id": SOURCE_ID, "page": 27, "evidence": "| Harness VLA (CC) | 94.0 | 80.0 | 88.0 | 90.0 | 87.0 | 87.0 | 71.0 | 62.0 | 82.4 |"}],
        },
        {
            "id": "adv_test_tab04",
            "split": "test",
            "primary_group": "table",
            "secondary_tags": ["Harness VLA", "Table 15", "LingBot-VLA"],
            "answerable": True,
            "question": "In Harness VLA Table 15 (page 53), what are the Optimizer, Learning rate, Vision encoder LR, Loss function, Chunk size, Flow steps, Global batch size, and Image resolution used for post-training LingBot-VLA on RoboTwin?",
            "expected_answer": "According to Harness VLA Table 15 (page 53), LingBot-VLA post-training on RoboTwin uses: Optimizer = AdamW, Learning rate = 1 x 10^-4, Vision encoder LR = 1 x 10^-6, Loss function = L1 Flow Matching (L1 FM), Chunk size = 50, Flow steps = 10, Global batch size = 256, and Image resolution = 224 x 224 (with top + wrist_left + wrist_right camera views).",
            "source": SOURCE_FILE,
            "pages": [53],
            "evidence_details": "Page 53, Table 15 ('Post-training configuration of LingBot-VLA on RoboTwin'): AdamW, LR 1x10^-4, Vision encoder LR 1x10^-6, L1 Flow Matching (L1 FM), Chunk size 50, Flow steps 10, Global batch size 256, Image resolution 224x224.",
            "required_facts": ["AdamW", "10^-4", "10^-6", "L1 Flow Matching", "50", "10", "256", "224"],
            "gold": [{"source_id": SOURCE_ID, "page": 53, "evidence": "Loss function: L1 Flow Matching (L1 FM)"}],
        },
        {
            "id": "adv_test_tab05",
            "split": "test",
            "primary_group": "table",
            "secondary_tags": ["Harness VLA", "Table 19", "Primitive Usage"],
            "answerable": True,
            "question": "In Harness VLA Table 19 (page 59), what are the exact call counts and percentages (%) of VLA_ACT and MOVE_TO across LIBERO, RoboTwin C2R, and RoboCasa365, and what is the total number of manipulation-primitive calls in each of the three environments?",
            "expected_answer": "According to Harness VLA Table 19 (page 59):\n- MOVE_TO: LIBERO = 6,263 (61.8%), RoboTwin C2R = 685 (40.9%), RoboCasa365 = 3,004 (38.7%).\n- VLA_ACT: LIBERO = 1,598 (15.8%), RoboTwin C2R = 794 (47.4%), RoboCasa365 = 2,746 (35.3%).\n- Total manipulation-primitive calls: LIBERO = 10,134 (100.0%), RoboTwin C2R = 1,675 (100.0%), and RoboCasa365 = 7,772 (100.0%).",
            "source": SOURCE_FILE,
            "pages": [59],
            "evidence_details": "Page 59, Table 19, rows 'MOVE_TO', 'VLA_ACT', and 'Total' across columns LIBERO, RoboTwin C2R, and RoboCasa365.",
            "required_facts": ["6263", "61.8", "1598", "15.8", "794", "47.4", "2746", "35.3", "10134", "1675", "7772"],
            "gold": [{"source_id": SOURCE_ID, "page": 59, "evidence": "| VLA_ACT | VLA | 1598 (15.8%) | 794 (47.4%) | 2746 (35.3%) |"}],
        },
        {
            "id": "adv_test_tab06",
            "split": "test",
            "primary_group": "table",
            "secondary_tags": ["AMP", "Table 3", "Single-Clip Imitation"],
            "answerable": True,
            "question": "In AMP Table 3 (page 73), what unit is used to measure imitation error, and on which two Humanoid motions does AMP (Ours) achieve a lower average pose error than Motion Tracking (give the exact values and dataset sizes for both)?",
            "expected_answer": "In AMP Table 3 (page 73), imitation performance is measured as average DTW-aligned pose error in units of meters (m) (± SD across 3 seeds x 32 episodes). The two Humanoid motions where AMP (Ours) achieves a lower pose error than Motion Tracking are:\n1) Humanoid 'Spin' (Dataset Size = 1.00s): AMP = 0.047 ± 0.002 m vs Motion Tracking = 0.063 ± 0.022 m (0.016 m lower).\n2) Humanoid 'Side-Flip' (Dataset Size = 2.44s): AMP = 0.124 ± 0.012 m vs Motion Tracking = 0.191 ± 0.043 m (0.067 m lower).",
            "source": SOURCE_FILE,
            "pages": [73],
            "evidence_details": "Page 73, AMP Table 3 caption (units: meters) and rows 'Humanoid | Spin' (1.00s, 0.063±0.022 vs 0.047±0.002) and 'Humanoid | Side-Flip' (2.44s, 0.191±0.043 vs 0.124±0.012).",
            "required_facts": ["meters", "Spin", "1.00", "0.047", "0.063", "Side-Flip", "2.44", "0.124", "0.191"],
            "gold": [{"source_id": SOURCE_ID, "page": 73, "evidence": "Note: AMP achieves lower pose error than Motion Tracking on Humanoid Spin (0.047m vs 0.063m) and Humanoid Side-Flip (0.124m vs 0.191m)."}],
        },

        # --- Group 2: FIGURE / DIAGRAM (5 cases: adv_test_fig01 .. adv_test_fig05) ---
        {
            "id": "adv_test_fig01",
            "split": "test",
            "primary_group": "figure_diagram",
            "secondary_tags": ["Harness VLA", "Figure 1 System Overview"],
            "answerable": True,
            "question": "In the Harness VLA Figure 1 system overview diagram (page 20), what are the three state variables and their exact vector notations shown inside the 'Robot State' box, and what is the 7-step command sequence shown in the horizontal strip below the planner?",
            "expected_answer": "In Harness VLA Figure 1 (page 20, crop p020_img02_Im2.png), the 'Robot State' box lists three variables: 1) eef_pos, [x,y,z], 2) eef_quat, [x,y,z,w], and 3) gripper_qpos, [q1,q2]. The horizontal command strip below the planner shows the 7-step sequence: Initial State -> VLA_ACT -> MOVE_TO -> SET_GRIPPER -> VLA_ACT -> MOVE_TO -> MOVE_TO.",
            "source": SOURCE_FILE,
            "pages": [20],
            "evidence_details": "Page 20, Figure 1 diagram crop p020_img02_Im2.png: 'Robot State' box (eef_pos [x,y,z], eef_quat [x,y,z,w], gripper_qpos [q1,q2]) and horizontal command strip (Initial State -> VLA_ACT -> MOVE_TO -> SET_GRIPPER -> VLA_ACT -> MOVE_TO -> MOVE_TO).",
            "required_facts": ["eef_pos", "eef_quat", "gripper_qpos", "Initial State", "VLA_ACT", "MOVE_TO", "SET_GRIPPER"],
            "gold": [{"source_id": SOURCE_ID, "page": 20, "evidence": "Initial State -> VLA_ACT -> MOVE_TO -> SET_GRIPPER -> VLA_ACT -> MOVE_TO -> MOVE_TO"}],
        },
        {
            "id": "adv_test_fig02",
            "split": "test",
            "primary_group": "figure_diagram",
            "secondary_tags": ["Harness VLA", "Figure 3 Failure Analysis"],
            "answerable": True,
            "question": "According to Harness VLA Figure 3 (page 29), how does the direct pi_RLinf baseline fail on the task-perturbed OBJECT-PRO variant (Obj-T) versus the swap-perturbed GOAL-PRO variant (Goal-S), and how does Harness VLA resolve both?",
            "expected_answer": "In Harness VLA Figure 3 (page 29, crops p029_img01_Im4.jpg..p029_img06_Im9.jpg), on the task-perturbed OBJECT-PRO variant (Obj-T), where the instruction redirects the target while the visual scene stays similar, pi_RLinf repeats the standard training behavior instead of following the new instruction. On the swap-perturbed GOAL-PRO variant (Goal-S), where the object layout is swapped, pi_RLinf blindly moves the object toward the old training-time region. Harness VLA resolves both by re-grounding the target through the agentic planner Pi, using analytic primitives for staging, and invoking VLA_ACT for the local contact-rich phase.",
            "source": SOURCE_FILE,
            "pages": [29],
            "evidence_details": "Page 29, Figure 3 caption and visual frames p029_img01_Im4.jpg..p029_img06_Im9.jpg comparing Obj-T and Goal-S rollouts.",
            "required_facts": ["repeats the standard behavior", "training-time region", "planner", "analytic primitives", "VLA_ACT"],
            "gold": [{"source_id": SOURCE_ID, "page": 29, "evidence": "under position-swap (Goal-S), pi_RLinf blindly moves the object toward the old training-time region after layout swap"}],
        },
        {
            "id": "adv_test_fig03",
            "split": "test",
            "primary_group": "figure_diagram",
            "secondary_tags": ["Harness VLA", "Figure 9 Flash Mode"],
            "answerable": True,
            "question": "In Harness VLA Figure 9 (page 35), what are the three bimanual manipulation procedures shown across the top row (VLA fullshot), middle row (Flash Mode), and bottom row (Flash Mode)?",
            "expected_answer": "In Harness VLA Figure 9 (page 35, crops p035_img01_Im53.jpg..p035_img09_Im61.jpg):\n1) Top row (VLA fullshot): grasp the container, hand it off between arms, and place it in the bin.\n2) Middle row (Flash Mode): hold the container, acquire the receiving bowl, and tilt the container to pour steel beads.\n3) Bottom row (Flash Mode): acquire the sponge while holding the plate, wipe the plate, and stow it after setting down the sponge.",
            "source": SOURCE_FILE,
            "pages": [35],
            "evidence_details": "Page 35, Figure 9 caption and rollout frame strips p035_img01_Im53.jpg..p035_img09_Im61.jpg.",
            "required_facts": ["hand it off", "pour steel beads", "sponge", "wipe the plate"],
            "gold": [{"source_id": SOURCE_ID, "page": 35, "evidence": "hold the container, acquire the receiving bowl, and tilt the container to pour steel beads."}],
        },
        {
            "id": "adv_test_fig04",
            "split": "test",
            "primary_group": "figure_diagram",
            "secondary_tags": ["AMP", "Figure 2 System Schematic"],
            "answerable": True,
            "question": "In the AMP Fig. 2 schematic overview diagram (page 65), what inputs feed into the 'Motion Prior' block, how is the total reward r_t formed, and what three signals enter the 'Policy' block?",
            "expected_answer": "In AMP Fig. 2 (page 65, crop p065_img01_Im3.png):\n1) The 'Motion Prior' block receives reference motions from the 'Dataset' block and the character state s_t from the 'Environment' block, and outputs the style-reward r_t^S.\n2) The total reward r_t is formed at the addition node (+) by summing the style-reward r_t^S (from Motion Prior) and the task-reward r_t^G (from Task): r_t = r_t^S + r_t^G.\n3) The 'Policy' block receives three input arrows: the state s_t (from Environment), the combined reward r_t (for training), and the goal g (from Task), and outputs action a_t to the Environment.",
            "source": SOURCE_FILE,
            "pages": [65],
            "evidence_details": "Page 65, Fig. 2 schematic diagram crop p065_img01_Im3.png and caption.",
            "required_facts": ["Dataset", "Environment", "s_t", "r_t^S", "r_t^G", "a_t"],
            "gold": [{"source_id": SOURCE_ID, "page": 65, "evidence": "An addition node (+) sums the style-reward r_t^S and task-reward r_t^G to form the combined scalar reward r_t = r_t^S + r_t^G."}],
        },
        {
            "id": "adv_test_fig05",
            "split": "test",
            "primary_group": "figure_diagram",
            "secondary_tags": ["FBRL", "Figure 1 Gridworld & Towers of Hanoi"],
            "answerable": True,
            "question": "In FBRL Figure 1 and Section 5 (pages 85–86), where are the robot start position and the 'Goal' cell located in the Gridworld diagram, what are the goal reward and per-step cost, and how is the 2-disc Towers of Hanoi state represented as a bit-string?",
            "expected_answer": "In FBRL Figure 1 (page 85, crops p085_img01_Im1.png and p085_img02_Im2.png) and Section 5 (pages 85–86): in the 3x3 Gridworld diagram, the robot is initialized in the bottom-left corner cell and the 'Goal' cell is in the top-right corner. In both Gridworld and Towers of Hanoi, the agent receives a reward of +1 upon reaching the goal and a step cost of -0.01 per time-step. In 2-disc Towers of Hanoi, the state is represented as a 6-bit string indicating which of the 3 pillars each disc is on (3 bits per disc, e.g., [1, 0, 0, 1, 0, 0] when the small disc is on the 1st pillar and the large disc is on the 1st/3rd pillar).",
            "source": SOURCE_FILE,
            "pages": [85, 86],
            "evidence_details": "Page 85 Figure 1 crops p085_img01_Im1.png (bottom-left robot, top-right 'Goal') & p085_img02_Im2.png, plus Page 85 Section 5.1 and Page 86 Section 5.2 text (reward 1, step cost -0.01, bit-string [1, 0, 0, 1, 0, 0]).",
            "required_facts": ["bottom-left", "top-right", "1", "-0.01", "[1, 0, 0, 1, 0, 0]"],
            "gold": [{"source_id": SOURCE_ID, "page": 85, "evidence": "Left diagram (p085_img01_Im1.png, 244x244 px): A 3x3 Gridworld where the robot icon is initialized in the bottom-left cell (start state) and the text 'Goal' is in the top-right cell"}],
        },

        # --- Group 3: CHART (5 cases: adv_test_cht01 .. adv_test_cht05) ---
        {
            "id": "adv_test_cht01",
            "split": "test",
            "primary_group": "chart",
            "secondary_tags": ["AWR", "Figure 4 Right Replay Buffer"],
            "answerable": True,
            "question": "In the right chart of AWR Figure 4 ('Replay Buffer: HalfCheetah-v2', page 9), what are the five replay buffer capacities compared in the legend, which capacity is marked as Default and achieves the highest return (~9,100), and which capacity performs worst (~4,200–4,500)?",
            "expected_answer": "In AWR Figure 4 right ('Replay Buffer: HalfCheetah-v2', page 9, crop p009_img03_Im13.png), the five replay buffer capacities in the legend are 5k (blue), 20k (orange), 50k (Default) (green), 100k (red), and 500k (purple). '50k (Default)' achieves the highest return (~9,100 at 2.0 x 10^7 samples), while the smallest buffer '5k' performs worst, plateauing around ~4,200–4,500 due to overfitting and training instability.",
            "source": SOURCE_FILE,
            "pages": [8, 9],
            "evidence_details": "Page 9, Figure 4 right crop p009_img03_Im13.png and Page 8 text: legend 5k, 20k, 50k (Default), 100k, 500k; 50k (Default) highest at ~9,100; 5k lowest at ~4,200-4,500.",
            "required_facts": ["5k", "20k", "50k", "100k", "500k", "9,100"],
            "gold": [{"source_id": SOURCE_ID, "page": 9, "evidence": "compares 5 replay buffer capacities: '5k' (blue), '20k' (orange), '50k (Default)' (green), '100k' (red), and '500k' (purple)."}],
        },
        {
            "id": "adv_test_cht02",
            "split": "test",
            "primary_group": "chart",
            "secondary_tags": ["AWR", "Figure 7 Static Off-Policy Bar Charts"],
            "answerable": True,
            "question": "In AWR Figure 7 (page 10), what 7 methods are compared on the X-axis of the static off-policy dataset bar charts, and on which two tasks is AWR (red bar) the highest-performing method overall (exceeding both the Demo dashed line and BEAR/BCQ)?",
            "expected_answer": "In AWR Figure 7 (page 10, crops p010_img03_Im21.png..p010_img06_Im24.png), the 7 methods on the X-axis are BC, PPO, TD3, SAC, BCQ, BEAR, and AWR (with a horizontal dashed line for 'Demo'). AWR is the highest-performing bar overall on two tasks: Ant-v2 (~4,550 vs Demo ~4,400 and BEAR ~3,400) and Walker2d-v2 (~4,480 vs Demo ~2,920, BCQ ~3,260, and BEAR ~3,080).",
            "source": SOURCE_FILE,
            "pages": [10],
            "evidence_details": "Page 10, Figure 7 bar chart crops p010_img03_Im21.png (Ant-v2) and p010_img06_Im24.png (Walker2d-v2).",
            "required_facts": ["BC", "PPO", "TD3", "SAC", "BCQ", "BEAR", "AWR", "Ant-v2", "Walker2d-v2"],
            "gold": [{"source_id": SOURCE_ID, "page": 10, "evidence": "X-axis categories (7 bars): BC (blue), PPO (orange), TD3 (green), SAC (purple), BCQ (brown), BEAR (pink), AWR (red)"}],
        },
        {
            "id": "adv_test_cht03",
            "split": "test",
            "primary_group": "chart",
            "secondary_tags": ["Harness VLA", "Figure 4a & 4b"],
            "answerable": True,
            "question": "In Harness VLA Figure 4 panels (a) LIBERO-PRO and (b) ROBOCASA365 (page 30), what are the exact cumulative task success rates (%) on LIBERO-PRO from k = 0 to k = 6 VLA invocations, and at what minimum k does each benchmark surpass its 'VLA full-shot' dashed baseline?",
            "expected_answer": "In Harness VLA Figure 4 (page 30, crops p030_img01_Im10.png and p030_img02_Im11.png):\n- On LIBERO-PRO (panel a), cumulative task success is: k=0 -> 3%, k=1 -> 52%, k=2 -> 68%, k=3 -> 73%, k=4 -> 77%, k=5 -> 79%, k=6 -> 80% (all invocations ceiling = 82%). It surpasses the 'VLA full-shot (50%)' baseline at k = 1 (52%).\n- On ROBOCASA365 (panel b), the 'VLA full-shot (30%)' baseline is first surpassed at k = 6 (where cumulative success reaches 31%, up from 29% at k=5, with all invocations ceiling = 48%).",
            "source": SOURCE_FILE,
            "pages": [30],
            "evidence_details": "Page 30, Figure 4(a) crop p030_img01_Im10.png (3, 52, 68, 73, 77, 79, 80; k=1 > 50%) and Figure 4(b) crop p030_img02_Im11.png (k=5 is 29%, k=6 is 31% > 30%).",
            "required_facts": ["3%", "52%", "68%", "73%", "77%", "79%", "80%", "k=1", "k=6", "31%"],
            "gold": [{"source_id": SOURCE_ID, "page": 30, "evidence": "Data points on the orange curve: k=0 -> 3%, k=1 -> 52%, k=2 -> 68%, k=3 -> 73%, k=4 -> 77%, k=5 -> 79%, k=6 -> 80%."}],
        },
        {
            "id": "adv_test_cht04",
            "split": "test",
            "primary_group": "chart",
            "secondary_tags": ["Harness VLA", "Figure 6 Task Completion Attribution"],
            "answerable": True,
            "question": "In Harness VLA Figure 6 ('Task completion attribution across benchmarks', page 32), what are the exact percentages of 'Analytic primitive finish' (blue) vs 'VLA primitive finish' (orange) across all 4 LIBERO suites, all 3 RoboCasa365 splits, and RoboTwin (50 tasks)?",
            "expected_answer": "In Harness VLA Figure 6 (page 32, crop p032_img01_Im25.png):\n- LIBERO: Spatial = 72% Analytic / 28% VLA; Object = 80% Analytic / 20% VLA; Goal = 79% Analytic / 21% VLA; LIBERO-10 = 84% Analytic / 16% VLA.\n- RoboCasa365: Composite seen = 40% Analytic / 60% VLA; Composite unseen = 17% Analytic / 83% VLA; Atomic = 14% Analytic / 86% VLA.\n- RoboTwin (50 tasks): 34% Analytic / 66% VLA.",
            "source": SOURCE_FILE,
            "pages": [32],
            "evidence_details": "Page 32, Figure 6 bar chart crop p032_img01_Im25.png: exact bar labels above each blue/orange bar.",
            "required_facts": ["72%", "28%", "80%", "20%", "79%", "21%", "84%", "16%", "40%", "60%", "17%", "83%", "14%", "86%", "34%", "66%"],
            "gold": [{"source_id": SOURCE_ID, "page": 32, "evidence": "Spatial = 72% Analytic / 28% VLA; Object = 80% Analytic / 20% VLA; Goal = 79% Analytic / 21% VLA; LIBERO-10 = 84% Analytic / 16% VLA."}],
        },
        {
            "id": "adv_test_cht05",
            "split": "test",
            "primary_group": "chart",
            "secondary_tags": ["AMP", "Figure 4 Target Heading Speed"],
            "answerable": True,
            "question": "In the right chart of AMP Fig. 4 ('Humanoid: Target Heading', page 70), what are the X-axis and Y-axis labels and units, and how do the 'Walk' (orange +), 'Run' (green triangle), and 'Locomotion' (blue circle) policies behave as Target Speed increases from 1 m/s to 5 m/s?",
            "expected_answer": "In AMP Fig. 4 right (page 70, crop p070_img02_Im13.png), the X-axis is 'Target Speed (m/s)' (from 0 to 5 m/s) and the Y-axis is 'Average Speed (m/s)' (from 0 to 5 m/s). As Target Speed increases from 1 to 5 m/s:\n- 'Walk' (orange +) stays flat around ~1.2–1.3 m/s and cannot reach higher target speeds.\n- 'Run' (green triangles) cannot match slow speeds—staying around ~2.8–3.0 m/s when Target Speed is 1–2.5 m/s—before rising to ~4.2 m/s at Target Speed 5 m/s.\n- 'Locomotion' (blue circles) closely follows the gray y=x diagonal dashed line from 1 m/s to 4.5 m/s by transitioning between walking (~1 m/s), jogging (~2.5 m/s), and running (~4.5 m/s).",
            "source": SOURCE_FILE,
            "pages": [69, 70],
            "evidence_details": "Page 70, Fig. 4 right crop p070_img02_Im13.png and Page 69-70 Section 8.2 text.",
            "required_facts": ["Target Speed (m/s)", "Average Speed (m/s)", "Walk", "Run", "Locomotion", "1.2", "2.8"],
            "gold": [{"source_id": SOURCE_ID, "page": 70, "evidence": "Scatter plot of X-axis 'Target Speed (m/s)' (from 0 to 5 m/s) vs Y-axis 'Average Speed (m/s)' (from 0 to 5 m/s)"}],
        },

        # --- Group 4: MULTI-HOP (4 cases: adv_test_mhp01 .. adv_test_mhp04) ---
        {
            "id": "adv_test_mhp01",
            "split": "test",
            "primary_group": "multi_hop",
            "secondary_tags": ["Harness VLA", "Table 7 & Page 61", "Flash Mode Speedup"],
            "answerable": True,
            "question": "Combining Harness VLA Table 7 (page 35) and Appendix G (pages 60–61), how many of the 80 LIBERO-Pro task identities have compiled Flash cards (and which two specific tasks have no successful source trace), how many total episodes out of 800 succeed in Flash Mode (max), and what is the execution time speedup factor of Flash Mode (max) compared to Codex (reasoning high) and Codex (no reasoning)?",
            "expected_answer": "Combining Table 7 (page 35) and Appendix G (pages 60–61):\n1) Out of 80 task identities in LIBERO-Pro, 78 have compiled Flash cards, while 2 tasks—'goal_swap_t0' and '10_swap_t9'—have no successful source trace and contribute 0/10.\n2) Aggregating all 800 episodes gives 581 successes (72.63% overall success rate).\n3) Flash Mode (max) has a mean execution time of 56.17s, giving a speedup factor of 9.42x over Codex (reasoning high) (529.23s / 56.17s = 9.42x, saving 473.06s) and 6.13x over Codex (no reasoning) (344.17s / 56.17s = 6.13x, saving 288.00s).",
            "source": SOURCE_FILE,
            "pages": [35, 60, 61],
            "evidence_details": "Multi-hop combining Page 35 Table 7 (Codex high = 78.50%, 529.23s; Codex no reasoning = 62.50%, 344.17s; Flash Mode max = 72.63%, 56.17s) and Pages 60-61 Appendix G (78 of 80 tasks have cards; goal_swap_t0 and 10_swap_t9 missing; 581/800 successes).",
            "required_facts": ["78", "goal swap t0", "10 swap t9", "581", "72.63", "56.17", "529.23", "344.17", "9.42", "6.13"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 35, "evidence": "| Flash Mode (max) | 79.0 | 70.0 | 86.0 | 93.0 | 74.0 | 65.0 | 60.0 | 54.0 | 72.63 | 56.17 |"},
                {"source_id": SOURCE_ID, "page": 60, "evidence": "Across the80task identities in LIBERO-Pro,78have cards; goal swap t0and10 swap t9have no successful source trace and contribute0/10. Aggregating all 800episodes gives581successes (72.63%)."},
            ],
        },
        {
            "id": "adv_test_mhp02",
            "split": "test",
            "primary_group": "multi_hop",
            "secondary_tags": ["Harness VLA", "Table 14 & Table 21 & Table 22", "GPT-6 Astra"],
            "answerable": True,
            "question": "Combining Harness VLA Table 14 (page 50) with Tables 21 and 22 (page 61), how many total rollouts are evaluated across LIBERO-Pro and RoboCasa365 combined, how many total successful rollouts does Codex (GPT-6 Astra) achieve across those two benchmarks combined, and how many percentage points does GPT-6 Astra gain over Codex (GPT-5.5) on RoboCasa365 Composite-Unseen?",
            "expected_answer": "Combining Table 14 (page 50) and Tables 21–22 (page 61):\n1) Total evaluated rollouts across LIBERO-Pro (800 rollouts) and RoboCasa365 (250 rollouts) combined = 1,050 rollouts.\n2) Codex (GPT-6 Astra) achieves 741/800 successes (92.63%) on LIBERO-Pro and 148/250 successes (59.20%) on RoboCasa365, for a combined total of 741 + 148 = 889 successful rollouts out of 1,050 (84.67%).\n3) On RoboCasa365 Composite-Unseen (Table 22), GPT-6 Astra achieves 42.50% vs Codex (GPT-5.5)'s 13.8%, gaining +28.70 percentage points (42.50% - 13.8% = 28.70%).",
            "source": SOURCE_FILE,
            "pages": [50, 61],
            "evidence_details": "Multi-hop combining Page 50 Table 14 (LIBERO-Pro = 800, RoboCasa365 = 250) and Page 61 Tables 21-22 (GPT-6 Astra LIBERO-Pro = 741/800 [92.63%], RoboCasa365 = 148/250 [59.20%], Composite-Unseen = 42.50% vs 13.8%).",
            "required_facts": ["1050", "889", "28.7"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 50, "evidence": "LIBERO-Pro"},
                {"source_id": SOURCE_ID, "page": 61, "evidence": "Table 21: LIBERO-Pro success rates (%) with GPT-6 Astra (741/800 successes = 92.63% overall)"},
            ],
        },
        {
            "id": "adv_test_mhp03",
            "split": "test",
            "primary_group": "multi_hop",
            "secondary_tags": ["Harness VLA", "Figure 6 & Table 20", "Call Share vs Terminal Finish Share"],
            "answerable": True,
            "question": "Combining Harness VLA Figure 6 (page 32) and Table 20 (page 60), compare the percentage of total manipulation calls that invoke VLA_ACT versus the percentage of successful rollouts whose final completion predicate finishes inside VLA_ACT on RoboTwin C2R (50 tasks) and on RoboCasa365 Atomic-Seen.",
            "expected_answer": "Combining Figure 6 (page 32) and Table 20 (page 60):\n- On RoboTwin C2R (50 tasks): VLA_ACT accounts for 47.4% of total manipulation-primitive calls (794 / 1,675 calls in Table 20), while 66% of successful rollouts finish with a VLA_ACT primitive (vs 34% Analytic primitive finish in Figure 6), a +18.6 percentage-point higher terminal finish share than call share.\n- On RoboCasa365: overall VLA_ACT call share across RoboCasa365 is 35.3% (2,746 / 7,772 calls in Table 20), whereas in Figure 6 the VLA primitive finish share reaches 86% on Atomic-Seen (and 83% on Composite-unseen, 60% on Composite-seen).",
            "source": SOURCE_FILE,
            "pages": [32, 59, 60],
            "evidence_details": "Multi-hop combining Page 32 Figure 6 (RoboTwin VLA finish = 66%, Analytic = 34%; RoboCasa365 Atomic VLA finish = 86%) and Page 59-60 Table 20 (VLA_ACT call share: RoboTwin C2R = 794 [47.4%], RoboCasa365 = 2746 [35.3%]).",
            "required_facts": ["47.4%", "66%", "35.3%", "86%"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 32, "evidence": "Middle panel (RoboCasa365): Composite seen = 40% Analytic / 60% VLA; Composite unseen = 17% Analytic / 83% VLA; Atomic = 14% Analytic / 86% VLA."},
                {"source_id": SOURCE_ID, "page": 59, "evidence": "VLA primitive (VLA_ACT): LIBERO = 1598 (15.8%), RoboTwin C2R = 794 (47.4%), RoboCasa365 = 2746 (35.3%)."},
            ],
        },
        {
            "id": "adv_test_mhp04",
            "split": "test",
            "primary_group": "multi_hop",
            "secondary_tags": ["AWR", "AMP", "Hyperparameters Comparison"],
            "answerable": True,
            "question": "Combining the AWR hyperparameters (page 16) and AMP Table 4 (page 78), compare the replay buffer size, discount factor gamma, TD(lambda) / GAE(lambda) parameter, and batch size used in AWR versus AMP.",
            "expected_answer": "Combining AWR hyperparameters (page 16) and AMP Table 4 (page 78):\n1) Replay buffer size: AWR uses a replay buffer capacity |D| = 50,000 (50k default), whereas AMP uses a Discriminator Replay Buffer Size B = 10^5 (100,000).\n2) Discount factor gamma: AWR uses gamma = 0.99; AMP uses gamma = 0.95 for Single-Clip Imitation and gamma = 0.99 for Tasks.\n3) TD(lambda) / GAE(lambda): both AWR and AMP use lambda = 0.95 (AMP sets both GAE(lambda) = 0.95 and TD(lambda) = 0.95).\n4) Batch size: both AWR and AMP use a batch size of 256 (AMP uses Batch Size = 256 and Discriminator Batch Size K = 256, with 4096 samples per update iteration).",
            "source": SOURCE_FILE,
            "pages": [16, 78],
            "evidence_details": "Multi-hop combining Page 16 (AWR hyperparameters: gamma=0.99, lambda=0.95, |D|=50,000, batch size=256) and Page 78 AMP Table 4 (B=10^5, gamma=0.95 single-clip / 0.99 tasks, TD(lambda)=GAE(lambda)=0.95, batch size=256).",
            "required_facts": ["50,000", "10^5", "0.99", "0.95", "256"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 16, "evidence": "discount factor gamma = 0.99, TD(lambda) parameter lambda = 0.95, replay buffer capacity |D| = 50,000"},
                {"source_id": SOURCE_ID, "page": 78, "evidence": "DiscriminatorStepsize = 10^-5 | Discriminator Replay Buffer Size B = 10^5" if False else "Discriminator Stepsize = 10^-5 | Discriminator Replay Buffer Size B = 10^5"},
            ],
        },

        # --- Group 5: TEXT (4 cases: adv_test_txt01 .. adv_test_txt04) ---
        {
            "id": "adv_test_txt01",
            "split": "test",
            "primary_group": "text",
            "secondary_tags": ["AWR", "Section 4 Related Work", "REPS & MPO"],
            "answerable": True,
            "question": "According to Section 4 of the AWR paper (page 6), how does AWR differ from REPS (Peters et al., 2010) in how the value function and sampling policy are modeled, and how does AWR differ from MPO (Abdolmaleki et al., 2018)?",
            "expected_answer": "According to page 6 (Section 4):\n1) Compared to REPS: REPS optimizes expected return rather than expected improvement, learns a linear value function by minimizing a complex dual function of the Bellman error (derived from a feature-matching constraint), and models the sampling policy using only the latest policy without incorporating experience replay from previous iterations. In contrast, AWR optimizes expected improvement, fits its value function V(s) with simple supervised regression on observed returns, and incorporates experience replay across past iterations.\n2) Compared to MPO: MPO applies a partial EM algorithm that fits a Q-function critic via bootstrapping with Retrace(lambda) off-policy correction, whereas AWR simply fits V(s) to observed returns in the replay buffer and performs weighted supervised regression on actions without fitting a bootstrapped Q-function.",
            "source": SOURCE_FILE,
            "pages": [6],
            "evidence_details": "Page 6, Section 4 paragraph 2: comparison of AWR with REPS (expected return, dual function of Bellman error, latest policy only without replay buffer) and MPO (partial EM, bootstrapped Q-function with Retrace(lambda)).",
            "required_facts": ["REPS", "dual function", "experience replay", "MPO", "Retrace"],
            "gold": [{"source_id": SOURCE_ID, "page": 6, "evidence": "MPO uses off-policy data for training a Q-function critic via bootstrapping and employs Retrace(λ) for off-policy correction"}],
        },
        {
            "id": "adv_test_txt02",
            "split": "test",
            "primary_group": "text",
            "secondary_tags": ["Harness VLA", "Appendix A & E", "File-Mediated REPL & Perception Isolation"],
            "answerable": True,
            "question": "According to Appendices A and E.2 of Harness VLA (pages 44 and 54), how does the synchronous file-mediated REPL protocol exchange commands and observations between the planner and worker, and what 6-step procedure must the agent follow to localize objects under perception isolation?",
            "expected_answer": "According to Appendix A (page 44, Table 8) and Appendix E.2 (page 54):\n- In the file-mediated REPL, at turn t the planner reads state_NN.json, RGB-D/world-map files, and memory, and writes one JSON primitive call to command.json. The environment worker consumes command.json, executes the primitive, and writes the next indexed state_(NN+1).json, perceptual files, log_(NN+1).json, and synchronization signal done_(NN+1).flag.\n- Under perception isolation (no privileged ground-truth object poses), the agent localizes entities in 6 steps: 1) Identify the relevant object/fixture/surface from RGB; 2) Pick pixels on its visible surface; 3) Index the matching precomputed world map at those pixels; 4) Sample multiple stable pixels and use a robust statistic (typically the median); 5) Avoid rims, object edges, table gaps, holes, reflections, and background pixels; 6) Re-localize whenever the robot, camera, object, base, fixture, or grasp state changes.",
            "source": SOURCE_FILE,
            "pages": [44, 54],
            "evidence_details": "Page 44 Appendix A (command.json, state_NN.json, log_NN.json, done_NN.flag) and Page 54 Appendix E.2 (6 numbered localization steps using RGB pixels, precomputed world map, and median).",
            "required_facts": ["command.json", "state", "done", "world map", "median", "Re-localize"],
            "gold": [{"source_id": SOURCE_ID, "page": 54, "evidence": "4. Sample multiple stable pixels and use a robust statistic, typically the median."}],
        },
        {
            "id": "adv_test_txt03",
            "split": "test",
            "primary_group": "text",
            "secondary_tags": ["AMP", "Appendix C", "Latent Space Model"],
            "answerable": True,
            "question": "According to Appendix C of the AMP paper (page 78), how is the Latent Space Model architecture structured and pretrained, what are the hidden layer sizes and latent encoding dimension of the encoder q(z_t | g_t), and how is it transferred to downstream tasks?",
            "expected_answer": "According to Appendix C (page 78):\n1) During pretraining, a Gaussian encoder q(z_t | g_t) = N(mu_q(g_t), Sigma_q(g_t)) maps a target pose goal g_t = (q_hat_{t+1}, q_hat_{t+2}) to a 16D latent variable z_t, which is passed to the low-level policy pi(a_t | s_t, z_t). They are trained jointly end-to-end with PPO and the reparameterization trick using a KL-regularizer (coefficient lambda = 10^-4) against a standard normal prior p_0(z_t) = N(0, I).\n2) The encoder network q consists of two hidden layers with 512 and 256 hidden units, followed by a linear output layer for mu_q and Sigma_q, with Latent Encoding Dimension = 16.\n3) For downstream tasks, the parameters of the low-level controller pi(a_t | s_t, z_t) are fixed (frozen), and a separate high-level controller u(z_t | s_t, g_t) is trained to output latent encodings z_t.",
            "source": SOURCE_FILE,
            "pages": [78],
            "evidence_details": "Page 78, Appendix C text and Table 5: encoder two hidden layers of 512 and 256 units, 16D latent encoding, KL-regularizer lambda = 10^-4, prior N(0, I), frozen pi with new high-level controller u(z_t | s_t, g_t).",
            "required_facts": ["512", "256", "16", "frozen", "high-level controller"],
            "gold": [{"source_id": SOURCE_ID, "page": 78, "evidence": "The encoder 𝑞 is modeled by a network consisting of two hidden layers, with 512 and 256 hidden units, followed by a linear output layer"}],
        },
        {
            "id": "adv_test_txt04",
            "split": "test",
            "primary_group": "text",
            "secondary_tags": ["FBRL", "Sections 5.1 & 5.2", "Backward Model Architecture"],
            "answerable": True,
            "question": "According to Sections 5.1 and 5.2 of the FBRL paper (pages 85–86), what are the exact neural network architectures, inputs, outputs, imagination steps, and asynchronous streams used for the backward model in Gridworld versus Towers of Hanoi?",
            "expected_answer": "According to Sections 5.1 and 5.2 (pages 85–86):\n1) In Gridworld (page 85): the backward model takes Cartesian coordinates (x, y) as input and predicts (Delta x, Delta y). Its architecture is a fully-connected network with 100 outputs followed by ReLU, followed by another fully-connected layer with 2 outputs (one for each state dimension). FBRL uses 10 steps of imagination with 1 asynchronous stream.\n2) In n-disc Towers of Hanoi (page 86): the backward model takes bit-strings indicating which pillar each disc is on (3n bits) and predicts a probability distribution for each bit over possible Delta values: P(Delta = -1), P(Delta = 0), P(Delta = 1). Its architecture is a fully-connected network with 100 outputs followed by ReLU, followed by a fully-connected layer with 9n outputs. FBRL uses 5 steps of imagination with 3 asynchronous streams.",
            "source": SOURCE_FILE,
            "pages": [85, 86],
            "evidence_details": "Page 85 Section 5.1 (Gridworld: inputs x,y -> Delta x, Delta y; 100 ReLU -> 2 outputs; 10 imagination steps, 1 stream) and Page 86 Section 5.2 (Towers of Hanoi: bit-strings -> P(Delta=-1,0,1); 100 ReLU -> 9n outputs; 5 imagination steps, 3 streams).",
            "required_facts": ["100", "RELU", "2 outputs", "10 steps", "1 asynchronous stream", "9n outputs", "5 steps", "3 asynchronous streams"],
            "gold": [
                {"source_id": SOURCE_ID, "page": 85, "evidence": "For FBRL, we used 10 steps of imagination with 1 asynchronous stream."},
                {"source_id": SOURCE_ID, "page": 86, "evidence": "For FBRL, we used 5 steps of imagination with 3 asynchronous streams."},
            ],
        },

        # --- Group 6: UNANSWERABLE (6 cases: adv_test_una01 .. adv_test_una06) ---
        {
            "id": "adv_test_una01",
            "split": "test",
            "primary_group": "unanswerable",
            "secondary_tags": ["AWR", "StarCraft Multi-Agent Challenge"],
            "answerable": False,
            "question": "In the AWR paper, what win rate (%) does AWR achieve on the StarCraft Multi-Agent Challenge (SMAC) 3s5z_vs_3s6z scenario?",
            "expected_answer": "Abstain / Insufficient evidence: The AWR paper (pages 1–19 of RL.pdf) only evaluates single-agent OpenAI Gym benchmarks, simulated humanoid/dog motion imitation, and static off-policy Gym datasets. It does not evaluate StarCraft, SMAC, or multi-agent RL scenarios.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by searching all 108 pages of RL.pdf for 'StarCraft', 'SMAC', and '3s5z': 0 matches.",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_test_una02",
            "split": "test",
            "primary_group": "unanswerable",
            "secondary_tags": ["Harness VLA", "CALVIN Benchmark"],
            "answerable": False,
            "question": "In the Harness VLA paper, what is the average completed task chain length (out of 5 sequential instructions) achieved by Harness VLA on the CALVIN ABC->D benchmark?",
            "expected_answer": "Abstain / Insufficient evidence: Harness VLA (pages 20–61 of RL.pdf) evaluates four benchmark families—standard LIBERO, LIBERO-Pro, RoboCasa365, and RoboTwin C2R (plus real-world dual-Franka demos)—and does not report any experiments or results on the CALVIN ABC->D benchmark.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by checking Table 14 (page 50) and searching RL.pdf for 'CALVIN' and 'ABC': no evaluation on CALVIN is reported.",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_test_una03",
            "split": "test",
            "primary_group": "unanswerable",
            "secondary_tags": ["Harness VLA", "LingBot-VLA GPU Hardware Count"],
            "answerable": False,
            "question": "According to Harness VLA Table 15 (page 53), how many NVIDIA H100 GPUs and how many total wall-clock training hours were used to post-train LingBot-VLA on RoboTwin?",
            "expected_answer": "Abstain / Insufficient evidence: While Harness VLA Table 15 (page 53) lists the optimizer (AdamW), learning rates, chunk size (50), global batch size (256), mixed precision (bf16/fp32), and FSDP2 distributed training for LingBot-VLA, it does not state the number of NVIDIA H100 GPUs or the wall-clock training hours.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by inspecting Page 52-53 (Appendix D.3 and Table 15) and searching RL.pdf for 'H100', 'A100', and GPU counts: not reported.",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_test_una04",
            "split": "test",
            "primary_group": "unanswerable",
            "secondary_tags": ["AMP", "Energy Consumption in Joules"],
            "answerable": False,
            "question": "In AMP Table 3 (page 73), what is the mechanical energy consumption in Joules per meter (J/m) for the 59 DoF T-Rex versus the 64 DoF Dog during trotting and walking?",
            "expected_answer": "Abstain / Insufficient evidence: AMP Table 3 (page 73) only reports the dataset size (in seconds) and the average DTW-aligned pose error (in meters) for T-Rex and Dog skills. It does not report mechanical energy consumption in Joules per meter (J/m).",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by inspecting Page 72-73 (AMP Section 8.4 and Table 3) and searching RL.pdf for 'Joule', 'J/m', and 'energy consumption': 0 matches.",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_test_una05",
            "split": "test",
            "primary_group": "unanswerable",
            "secondary_tags": ["FBRL", "5-Disc Towers of Hanoi"],
            "answerable": False,
            "question": "In FBRL Figure 3 (page 87), how many training steps does FBRL take to converge on the 5-disc (n = 5) Towers of Hanoi task with a horizon of 300 steps?",
            "expected_answer": "Abstain / Insufficient evidence: The FBRL paper (pages 86–87, Section 5.2 and Figure 3) only evaluates Towers of Hanoi for n = 2 discs (horizon 50) and n = 3 discs (horizon 100). It does not evaluate n = 5 discs or a 300-step horizon.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by inspecting Page 86 Section 5.2 and Page 87 Figure 3: only n=2 and n=3 discs are evaluated.",
            "required_facts": [],
            "gold": [],
        },
        {
            "id": "adv_test_una06",
            "split": "test",
            "primary_group": "unanswerable",
            "secondary_tags": ["AMP", "Real-World Quadruped Robot Hardware"],
            "answerable": False,
            "question": "In the AMP paper (pages 62–81), what was the real-world sim-to-real transfer success rate (%) when deploying the 64 DoF dog policy onto a physical Unitree A1 quadruped robot?",
            "expected_answer": "Abstain / Insufficient evidence: All experiments in the AMP paper (pages 62–81, using the Bullet physics engine) are conducted entirely on physically simulated characters (Humanoid, T-Rex, and Dog); the paper does not deploy policies onto a physical Unitree A1 or any real-world quadruped robot.",
            "source": SOURCE_FILE,
            "pages": [],
            "evidence_details": "Verified by inspecting AMP Sections 8-9 (pages 68-73) and searching RL.pdf for 'Unitree', 'sim-to-real', and 'quadruped hardware': 0 matches.",
            "required_facts": [],
            "gold": [],
        },
    ]
    return cases


def main() -> None:
    pdf_path = DAY7_ROOT / SOURCE_FILE
    out_dir = DAY7_ROOT / "data" / "advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks, manifest = build_chunks_and_manifest(pdf_path)
    cases = build_benchmark()

    # Validate answerable benchmark questions against chunks using starter/baseline.py validate()
    answerable_questions = [
        {
            "id": c["id"],
            "split": c["split"],
            "question": c["question"],
            "expected_answer": c["expected_answer"],
            "gold": c["gold"],
        }
        for c in cases
        if c["answerable"]
    ]
    validate(chunks, answerable_questions)

    (out_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "benchmark.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dev_cases = [c for c in cases if c["split"] == "dev"]
    test_cases = [c for c in cases if c["split"] == "test"]
    print(f"Built {len(chunks)} multimodal chunks across {manifest['document']['unique_indexed_pages']} unique pages.")
    print("Modality breakdown:", manifest["index_statistics"]["by_modality"])
    print(f"Benchmark validated: {len(dev_cases)} dev cases, {len(test_cases)} test cases.")


if __name__ == "__main__":
    main()
