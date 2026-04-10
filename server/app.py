# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uvicorn
from openenv.core.env_server.http_server import create_app
from drone_env.drone_environment import DroneEnvironment, DroneAction, DroneObservation

def make_env():
    try:
        from drone_env.drone_environment import DroneEnvironment
        return DroneEnvironment()
    except Exception as e:
        import traceback
        print(f"DEBUG: make_env failed with: {e}")
        traceback.print_exc()
        raise e

app = create_app(
    env=make_env,
    action_cls=DroneAction,
    observation_cls=DroneObservation,
    env_name="drone_env"
)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
