
# Optimizing Humanoid Gait with Arm Articulation and Extended Observations in Deep RL

<!-- 项目核心信息徽章 -->
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-FF6F00)](https://arxiv.org/abs/1707.06347)

<!-- 展示图：climb_down -->
![climb_down](https://github.com/user-attachments/assets/0cbca7ab-ade9-4e77-9b5c-9d2fafead47f)

## Introduction

This repository contains code for training humanoid robots to run using Deep Reinforcement Learning (Deep RL). It focuses on **optimizing gait by incorporating arm articulation and extending observation spaces**.

This project is a fork and modification of [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking), extending it with significant improvements in upper-body control and dynamic running capabilities.

## Tech Stack & Dependencies

This project relies on the following powerful libraries and frameworks:

<!-- 技术栈徽章区域 -->
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo_2.2.0-8A2BE2)](https://mujoco.org/)
[![Ray](https://img.shields.io/badge/Distributed-Ray_1.9.2-028CF0?logo=ray&logoColor=white)](https://docs.ray.io/en/releases-1.9.2/)
[![NumPy](https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-11557c)](https://matplotlib.org/)

## Key Features & Contributions

Unlike standard humanoid locomotion tasks that often freeze or ignore upper body dynamics, this project introduces:

1.  **Arm Articulation**: Added **14 new arm joints**, increasing the observation dimension from **37 to 65**.
2.  **Extended Behaviors**:
    *   Running using only legs.
    *   Using hands for active balance.
    *   Running with synchronized hand and leg movements.
3.  **Reward Engineering**: Added specific reward functions related to arm movements to support the training of robotic arm operations during locomotion.

## Directory Structure

A rough outline for the repository structure:

```text
LearningHumanoidRunning/
├── envs/                # Actions, observation space, PD gains, simulation steps, init...
├── tasks/               # Reward functions, termination conditions
├── rl/                  # PPO implementation (PyTorch), actor/critic networks, normalization
├── models/              # MuJoCo model files (XMLs, meshes, textures)
├── trained/             # Pre-trained models (e.g., for JVRC)
└── scripts/             # Utility scripts for debugging and plotting
```

## Installation

### Prerequisites

*   **Python**: 3.7.11
*   **OS**: Linux / macOS (Recommended for MuJoCo)

### Setup

Install the required packages using pip:

```bash
# Core dependencies
pip install torch
pip install ray==1.9.2
pip install transforms3d
pip install matplotlib
pip install scipy

# Simulation dependencies
pip install mujoco==2.2.0
pip install mujoco-python-viewer
```

## Usage

### Supported Environments

The following environments are available for training and testing:

| Task Description | Environment Name |
| :--- | :--- |
| Basic Walking Task | `jvrc_walk` |
| Stepping Task (Precision stepping using footsteps) | `jvrc_step` |
| Walking Task (incorporating arm movement) | `jvrc_arm` |
| Run Task (Legs only) | `jvrc_run` |
| Run Task (Using both legs and arms) | `jvrc_run_arm` |

### Training

To train a policy from scratch, use the `run_experiment.py` script.

**Arguments:**
*   `--logdir`: Path to save experiment logs and checkpoints.
*   `--num_procs`: Number of CPU processes to use for parallel sampling (Ray).
*   `--env`: The environment ID (see table above).

**Example:**
```bash
python run_experiment.py train \
    --logdir ./logs/exp_run_arm_v1 \
    --num_procs 16 \
    --env jvrc_run_arm
```

### Visualization & Playback

To visualize a trained policy or debug the environment, you can use the scripts in the `scripts/` directory.

**Example (Debugging the Stepper task):**

```bash
PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path ./logs/exp_run_arm_v1
```

## What you should see (Demo)

Here is a demonstration of the trained policy in action:

<video src="https://github.com/user-attachments/assets/08628f41-29f4-463e-947a-f9cd4d0b210c" controls="controls" style="max-width: 100%;">
</video>

## License

This project is licensed under the **BSD-2-Clause License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   Original codebase: [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking)
*   Physics engine: [MuJoCo](https://mujoco.org/)
