---
title: Agentic Drone PID Tuner
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Agentic Drone PID Tuner

## Motivation and Real-World Use Case

Autonomous systems, ranging from delivery drones to industrial robotics and aerospace vehicles, depend heavily on tightly tuned Proportional-Integral-Derivative (PID) controllers to maintain stability in complex environments. However, traditional PID tuning relies heavily on human-in-the-loop manual iteration or computationally expensive heuristic searches across narrow operational envelopes.

The **Agentic Drone PID Tuner** introduces an OpenEnv RL playground designed to train LLM-driven agents to automate this process. Rather than acting strictly as a static grid search, AI agents in this environment must assess state observations (Root-Mean-Square Error and target oscillation) over short duration physics simulations, adjust against turbulence and dynamic payloads, and issue corrected continuous PID gains.

If an AI can reason about dynamic disturbances (wind, changing payload mass) and adjust physics controllers autonomously without heuristic blind spots, we unlock massive efficiency gains in industrial autonomy deployment.

## Environment Details

This environment simulates a 1D vertical drone flight subject to turbulent drag, payload scaling and base physics equations. The agent's goal is to tune the PID parameters to securely reach and maintain a targeted altitude.

*   **Action Space:** `DroneAction` requires 3 continuous parameter adjustments.
    *   `Kp` (float): Proportional gain — acts on the immediate target error.
    *   `Ki` (float): Integral gain — acts on the accumulated historical error.
    *   `Kd` (float): Derivative gain — anticipates future error by tracking the error rate of change.

*   **Observation Space:** `DroneObservation` gives the agent immediate post-step feedback.
    *   `rmse` (float): The root-mean-square error of the flight path versus the `target_altitude`.
    *   `oscillation` (float): The total number of times the drone crossed the target setpoint plane (over-correction penalty metric).
    *   `reward` (float): Shaped gradient reward for reinforcement backpropagation.

## Evaluation Tasks

We implement domain randomization during episode generation to prevent trivial "gaming" of the gradient. Agents are scored deterministically by a non-exploitable grader relying on strictly bounded exponential decay over the RMSE penalty.

1.  **Easy (`stability_easy`)**: Standard 1.0kg drone against mild -5N wind. Reach 50m. Tests basic capability of converging on stable PID weights.
2.  **Medium (`payload_medium`)**: Drone mass doubled to 2.0kg and wind resistance increased to -10N. Reach 100m. Tests the agent's ability to boost integral power against heavy inertial momentum without crashing. 
3.  **Hard (`stormy_hard`)**: Requires extreme finesse. An unnaturally light drone (0.5kg) against severe -15N downdrafts. Reach 150m. Without tight derivative predictions, the drone's low inertia causes it to be violently flung past the goal line.

## Baseline Scores

Zero-shot evaluations on untuned base LLM models (e.g. `llama3` baseline running inference with generic starting prompts and 5 steps) yield the following approximate benchmark scores (scaled 0.0 - 1.0, 1.0 being flawless 0-RMSE hold):

*   `stability_easy`: **~0.42**
*   `payload_medium`: **~0.25**
*   `stormy_hard`: **~0.08**

Fine-tuned agents using GRPO methodologies should easily achieve `>0.8` scores on target evaluations.

## Quick Start

1.  **Install the Environment:**
    Navigate to the `envs/drone_env` directory and install it in editable mode:
    ```bash
    cd envs/drone_env
    pip install .
    ```

2.  **Start the OpenEnv Server:**
    Run the following command to start the FastAPI server (using Uvicorn):
    ```bash
    python -m uvicorn drone_env.server:app --host 0.0.0.0 --port 8000
    ```
    The server will be available at `http://127.0.0.1:8000`.

3.  **Run the Inference Example:**
    Open a new terminal, verify your environment settings, and run:
    ```bash
    export HF_TOKEN="your_token_here"
    python inference.py
    ```

## Technical Details

*   **Environment:** `drone_env/drone_environment.py`
    *   **Action Space:** `DroneAction` (`Kp`, `Ki`, `Kd` floats).
    *   **Observation Space:** `DroneObservation` (`rmse`, `oscillation`, `reward` floats).
*   **Physics Simulation:** `drone_env/kinematics.py`
*   **OpenEnv Server:** `drone_env/server.py`

The environment leverages predefined configurations in `openenv.yaml` to score agents based on altitude targeting and stability against varying mass and wind.
