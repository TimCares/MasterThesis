import os
import PIL.Image
import io
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import urllib
from datasets.utils.file_utils import get_datasets_user_agent
from torchvision.datasets.utils import download_url

USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, img_path, idx):
    for _ in range(3):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=5) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                w, h = image.size
                if w==1 or h==1:
                    return
                path = os.path.join(img_path, f"{idx}.jpg")
                image.save(path, format='JPEG')
            break
        except Exception:
            pass


def main():
    path_to_data = os.path.join('/workspace', "conceptual_captions_12m")
    img_path = os.path.join(path_to_data, "images")
    os.makedirs(path_to_data, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    download_url(url='https://storage.googleapis.com/conceptual_12m/cc12m.tsv', root=path_to_data)
    index_path = os.path.join(path_to_data, "cc12m.tsv") 
    index = pd.read_csv(index_path, sep='\t', header=None).reset_index(drop=True)
    index.columns = ['image_url', 'caption']
    already_existing = os.listdir(img_path)
    already_existing = [int(os.path.splitext(img)[0]) for img in already_existing]
    index = index[~index.index.isin(already_existing)][:int(3e6)]
    n_workers = os.cpu_count()*4
    with concurrent.futures.ThreadPoolExecutor(n_workers) as executor:
        futures = [executor.submit(fetch_single_image, url, img_path, idx) for idx, url in 
                   tqdm(index['image_url'].items(), total=len(index), desc="Scheduling tasks")]
        list(tqdm(concurrent.futures.as_completed(futures), total=len(index), desc="Downloading images"))

    n_failed = len(index) - len(os.listdir(img_path))
    print(f"Failed to download {n_failed} images (pairs). Percentage: {n_failed/len(index)*100:.2f}%")
    
if __name__ == "__main__":
    main()