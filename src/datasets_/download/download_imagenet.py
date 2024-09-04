from huggingface_hub import snapshot_download
snapshot_download(repo_id="ILSVRC/imagenet-1k", repo_type="dataset", allow_patterns=["train_images_*", "val_images_*"], local_dir='/workspace/imagenet')
