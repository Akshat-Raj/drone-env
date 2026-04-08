# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._previous_error = 0
        self._integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._previous_error) / dt
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self._previous_error = error
        return output

class Drone1D:
    def __init__(self, mass, target_altitude, Kp, Ki, Kd):
        self.mass = mass
        self.g = 9.81
        self.position = 0
        self.velocity = 0
        self.pid = PIDController(Kp, Ki, Kd, setpoint=target_altitude)
        self.wind_force = 0

    def update_wind(self, base_wind, turbulence_std, dt):
        # Ornstein-Uhlenbeck process for stochastic wind
        theta = 0.1  # Mean reversion rate
        mu = base_wind    # Mean wind force
        sigma = turbulence_std # Volatility
        dw = np.random.normal(0, np.sqrt(dt))
        self.wind_force += theta * (mu - self.wind_force) * dt + sigma * dw

    def step(self, dt):
        thrust = self.pid.update(self.position, dt)
        # Action clipping for safety
        thrust = np.clip(thrust, 0, self.mass * self.g * 2)
        
        net_force = thrust - self.mass * self.g + self.wind_force
        acceleration = net_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Ground constraint
        if self.position < 0:
            self.position = 0
            self.velocity = 0
