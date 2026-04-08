# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Dict, Any, Optional, Type
from abc import ABC, abstractmethod

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel

from .kinematics import Drone1D

class DroneAction(Action):
    Kp: float
    Ki: float
    Kd: float

class DroneObservation(Observation):
    rmse: float
    oscillation: float
    reward: float

class DroneEnvironment(Environment[DroneAction, DroneObservation, Any]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drone: Optional[Drone1D] = None
        self.target_altitude: float = 0
        self.base_wind: float = 0
        self.history = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        params: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ) -> DroneObservation:
        if seed is not None:
            np.random.seed(seed)

        # Domain Randomization or Specific Task Parameters (for Graders)
        mass = params.get("mass", np.random.uniform(0.5, 2.0)) if params else np.random.uniform(0.5, 2.0)
        self.target_altitude = params.get("target_altitude", np.random.uniform(20, 150)) if params else np.random.uniform(20, 150)
        self.base_wind = params.get("base_wind", np.random.uniform(-15, -5)) if params else np.random.uniform(-15, -5)
        
        # Initial drone setup
        self.drone = Drone1D(mass=mass, target_altitude=self.target_altitude, Kp=0, Ki=0, Kd=0)
        self.history = []
        
        # Initial observation is zero state
        return DroneObservation(rmse=0, oscillation=0, reward=0)

    def step(self, action: DroneAction) -> DroneObservation:
        if self.drone is None:
            raise RuntimeError("Environment must be reset before stepping.")

        # Action Clipping (Safety Firewall)
        Kp = np.clip(action.Kp, 0, 10)
        Ki = np.clip(action.Ki, 0, 5)
        Kd = np.clip(action.Kd, 0, 10)

        self.drone.pid.Kp = Kp
        self.drone.pid.Ki = Ki
        self.drone.pid.Kd = Kd

        dt = 0.02  # 50Hz simulation
        num_steps = 500  # 10-second simulation
        
        positions = []
        for _ in range(num_steps):
            self.drone.update_wind(base_wind=self.base_wind, turbulence_std=0.5, dt=dt)
            self.drone.step(dt)
            positions.append(self.drone.position)
        
        positions = np.array(positions)
        errors = positions - self.target_altitude
        rmse = float(np.sqrt(np.mean(errors**2)))
        
        # Oscillation is measured as the number of times the drone crosses the setpoint
        crossings = np.where(np.diff(np.sign(errors)))[0]
        oscillation = float(len(crossings))

        # Reward Shaping
        # Reward is 1.0 for perfect flight (RMSE=0), and decays exponentially
        reward = float(np.exp(-0.1 * rmse))

        obs = DroneObservation(rmse=rmse, oscillation=oscillation, reward=reward)
        
        self.history.append({
            "action": action.dict(),
            "rmse": rmse,
            "oscillation": oscillation,
            "reward": reward
        })

        return obs

    def render(self) -> Any:
        # For a real application, you might render a plot of the altitude vs. time
        return self.history[-1] if self.history else {}

    def state(self) -> Dict[str, Any]:
        """Return the current state of the environment."""
        if self.drone is None:
            return {"status": "uninitialized"}
        return {
            "position": self.drone.position,
            "velocity": self.drone.velocity,
            "target_altitude": self.target_altitude,
            "base_wind": self.base_wind,
            "mass": self.drone.mass,
            "Kp": self.drone.pid.Kp,
            "Ki": self.drone.pid.Ki,
            "Kd": self.drone.pid.Kd,
        }

    def close(self) -> None:
        print("Environment closed.")
