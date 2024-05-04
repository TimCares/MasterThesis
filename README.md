# MasterThesis
Thesis on Multimodal Representation Learning using Knowledge-Distillation with Data2Vec
- Work in Progress -

### Runpod.io

Upload data to network volume: 

```bash
scp -P <tcp-port> -i ~/.ssh/<private ssh key> -r /path/to/local/dir root@xxx.xxx.xxx.xxx:/workspace
```

### Download Imagnet

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```

```bash
sudo apt-get install git-lfs
```

```bash
pip install -U "huggingface_hub[cli]"
```

```bash
huggingface-cli login
```

```bash
git clone https://huggingface.co/datasets/imagenet-1k
```

If not automatically all is downloaded:
```bash
git lfs pull --include="<path/to/file1>,<path/to/file2>"
```

Example:
```bash
git lfs pull --include="data/train_images_0.tar.gz,data/train_images_1.tar.gz,data/train_images_2.tar.gz,data/val_images.tar.gz"
```
