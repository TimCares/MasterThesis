import os
import shutil
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str)
    args = parser.parse_args()

    get_type = lambda x: os.path.splitext(os.path.basename(x))[0].split('-')[-1]
    base_path = args.model_ckpt
    model_paths = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.ckpt')]
    output_paths = [os.path.join(base_path, f'fp32_last_{get_type(v)}.ckpt') for v in model_paths]
    for model_path, output_path in zip(model_paths, output_paths):
        convert_zero_checkpoint_to_fp32_state_dict(model_path, output_path)

if __name__ == "__main__":
    main()