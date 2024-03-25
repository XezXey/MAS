import numpy as np
import torch as th
import subprocess, os
import argparse

"""
Example: 
python  -m sample.mas 
        --model_path save/nba/nba_diffusion_model/checkpoint_500000.pth 
        --num_samples 10 
        --seed 0 
        --output_dir /data/mint/CR7_data/motion_diffusion/MAS/nba/
"""

def run_mas(model_path, num_samples, seed, output_dir):
    subprocess.run([f"python", "-m", "sample.{args.mode}",
                    "--model_path", model_path,
                    "--num_samples", str(num_samples),
                    "--seed", str(seed),
                    "--output_dir", output_dir])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--seed', nargs='+', type=int, default=[0])
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--mode', type=str, choices=['generate', 'mas'], default='mas')
    
    args = parser.parse_args()
    print(f'Running MAS with seed = {args.seed}')
    for seed in args.seed:
        run_mas(args.model_path, args.num_samples, seed, args.output_dir)
    print('MAS done!')
