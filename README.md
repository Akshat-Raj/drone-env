---
title: Agentic Drone PID Tuner
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Agentic Drone PID Tuner

This OpenEnv environment implements the **Agentic Drone PID Tuner** described in 
the project overview. It provides a simulated 1D drone that an LLM agent can learn to control by tuning PID coefficients.                                         
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
