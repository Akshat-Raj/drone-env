# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from openenv.core.generic_client import GenericEnvClient
import numpy as np
import re
import os
from openai import OpenAI

# --- TRL and GRPO Setup ---
def get_reward(rmse):
    """Calculates reward based on RMSE."""
    return float(np.exp(-0.1 * rmse))

def extract_pid_from_text(text):
    """Extracts PID values from the LLM's output using regex."""
    try:
        p_match = re.search(r"P:\s*([\d\.]+)", text)
        i_match = re.search(r"I:\s*([\d\.]+)", text)
        d_match = re.search(r"D:\s*([\d\.]+)", text)
        
        p = float(p_match.group(1)) if p_match else 0.0
        i = float(i_match.group(1)) if i_match else 0.0
        d = float(d_match.group(1)) if d_match else 0.0
        
        return {"Kp": p, "Ki": i, "Kd": d}
    except (AttributeError, ValueError):
        # Return default values if parsing fails
        return {"Kp": 1.0, "Ki": 0.1, "Kd": 0.5}

def run_grpo_tuning():
    """
    Main function to run the GRPO training loop.
    This is a simplified example and would require a more robust implementation for real train
ing.                                                                                          
    """
    # 1. Initialize Environment Client
    # NOTE: The environment server might be running on localhost during proxy evaluation. We check OPENENV_BASE_URL if it's separate from API_BASE_URL. 
    env_base_url = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:7860")
    try:
        async_client = GenericEnvClient(base_url=env_base_url)
        client = async_client.sync()
    except Exception as e:
        print(f"Failed to connect to environment server at {env_base_url}: {e}")
        return

    # 2. Load a Pre-trained LLM and Tokenizer
    model_name = "gpt2" # Using gpt2 for simplicity, but a larger model is recommended
    
    # Initialize OpenAI client using competition injected variables
    llm_api_base = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy_key"))
    llm_model = os.environ.get("MODEL_NAME", "llama3")
    
    openai_client = OpenAI(
        base_url=llm_api_base,
        api_key=llm_api_key
    )
    
    # Run through the tasks defined in openenv.yaml
    tasks = [
        {"id": "stability_easy", "target_altitude": 50.0, "mass": 1.0, "base_wind": -5.0},
        {"id": "payload_medium", "target_altitude": 100.0, "mass": 2.0, "base_wind": -10.0},
        {"id": "stormy_hard", "target_altitude": 150.0, "mass": 0.5, "base_wind": -15.0}
    ]

    for task_info in tasks:
        task_id = task_info["id"]
        target_altitude = task_info["target_altitude"]
        base_wind = task_info["base_wind"]
        
        print(f"[START] task={task_id}", flush=True)

        try:
            client.reset(params=task_info)
        except Exception as e:
            print(f"Error during reset: {e}")
            continue

        # run 5 tuning episodes for each task
        best_reward = 0.0
        for episode in range(5):
            # a. Create the Prompt
            prompt_text = f"Mission: Reach {target_altitude:.0f}m altitude. Wind is {base_wind:.0f}N. Predict PID gains. Format: P: X.XX, I: Y.YY, D: Z.ZZ"

            # b. Generate completions using OpenAI Proxy instead of local model
            try:
                response = openai_client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a PID tuning expert. Provide PID gains."},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=50,
                    temperature=0.7
                )
                text = response.choices[0].message.content
            except Exception as e:
                print(f"Error calling LLM: {e}")
                text = ""
            
            pid_params = extract_pid_from_text(text)
            
            # c. Evaluate step
            try:
                obs = client.step(pid_params)
                reward = get_reward(obs.rmse)
            except Exception as e:
                print(f"Error during step: {e}")
                reward = 0.0
                
            print(f"[STEP] step={episode+1} reward={reward:.4f}", flush=True)
            
            if reward > best_reward:
                best_reward = reward

        # Ensure the score is strictly between 0 and 1 as required by the validator
        final_score = max(0.001, min(0.999, best_reward))
        print(f"[END] task={task_id} score={final_score:.4f} steps=5", flush=True)

    print("\nGRPO Tuning Complete.", flush=True)

# The environment variables must be defined as per Round 1 rules.
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3-70b-8192")
HF_TOKEN = os.environ.get("HF_TOKEN") 
API_KEY = os.environ.get("OPENAI_API_KEY")

# Task Definitions (Match openenv.yaml)
TASK_DEFINITIONS = {
    "drone_mission": {
        "mission": "drone_mission",
        "params": {
            "altitude": {"type": "float", "min": 0, "max": 1000},
            "wind": {"type": "float", "min": -100, "max": 100},
        },
    },
}

if __name__ == "__main__":
    print("Starting Agentic Drone PID Tuner Inference...")
    print("NOTE: Ensure the OpenEnv server is running for 'drone_env'.")
    print("`openenv-server --env drone_env`")
    
    # This example focuses on the GRPO loop. A full implementation would
    # separate training and inference more cleanly.
    run_grpo_tuning()
