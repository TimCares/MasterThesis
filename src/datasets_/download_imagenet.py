from datasets import load_dataset

if __name__ == "__main__":
    data_files = ['data/train_images_0.tar.gz', 'data/train_images_1.tar.gz', 'data/train_images_2.tar.gz', 'data/val_images.tar.gz']
    dataset = load_dataset("imagenet-1k", data_dir='../data/', data_files='data/val_images.tar.gz')