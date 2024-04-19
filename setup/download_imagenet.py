from datasets import load_dataset

if __name__ == "__main__":
    load_dataset("imagenet-1k", cache_dir="/workspace/huggingface")

#$ export HF_DATASETS_CACHE="/workspace/huggingface"