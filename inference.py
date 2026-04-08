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
    This is a simplified example and would require a more robust implementation for real training.
    """
    # 1. Initialize Environment Client
    async_client = GenericEnvClient(base_url="http://127.0.0.1:8000")
    client = async_client.sync()
    client.reset()

    # 2. Load a Pre-trained LLM and Tokenizer
    model_name = "gpt2" # Using gpt2 for simplicity, but a larger model is recommended
    
    # Using GRPOTrainer and GRPOConfig as per modern TRL
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. The "Wordle" Pattern Training Loop
    for episode in range(5): # Run 5 tuning episodes
        # Get mission details from the environment's reset state
        target_altitude = 100 # Default if info not available
        base_wind = -10 # Default if info not available

        # a. Create the Prompt
        prompt_text = f"Mission: Reach {target_altitude:.0f}m altitude. Wind is {base_wind:.0f}N. Predict PID gains."
        query_tensor = tokenizer.encode(prompt_text, return_tensors="pt")

        # b. Generate multiple completions (the "Group" in GRPO)
        generation_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "num_return_sequences": 4,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # Generation with the base model for demonstration
        outputs = model.generate(query_tensor, **generation_kwargs)
        response_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # c. Evaluate each completion in the environment
        rewards = []
        actions = []
        for text in response_texts:
            pid_params = extract_pid_from_text(text)
            actions.append(pid_params)
            
            obs = client.step(pid_params)
            reward = get_reward(obs.rmse)
            rewards.append(reward)

        print(f"Episode {episode+1}: Mean Reward: {np.mean(rewards):.4f}")

    print("\nGRPO Tuning Complete.")
    # You would save the fine-tuned model here
    # model.save_pretrained("tuned_drone_controller")
    # tokenizer.save_pretrained("tuned_drone_controller")

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
