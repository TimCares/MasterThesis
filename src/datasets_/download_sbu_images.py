import os
import PIL.Image
import io
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import urllib
from datasets.utils.file_utils import get_datasets_user_agent
from torchvision.datasets.utils import download_url
import tarfile
import json

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
    path_to_data = os.path.join('/workspace', "sbu")
    img_path = os.path.join(path_to_data, "images")
    os.makedirs(path_to_data, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    index_name = "sbu-captions-all.json"
    index_path = os.path.join(path_to_data, index_name)

    if not os.path.exists(index_path):
        url="https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz"
        download_url(url=url, root=path_to_data)
        filepath = os.path.join(path_to_data, os.path.basename(url))
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(path=path_to_data)
        os.remove(filepath)

    with open(index_path) as f:
        sbu = json.load(f)

    sbu.pop('user_ids', None)
    sbu = pd.DataFrame(sbu)
    index = sbu['image_urls'].str.split('/').str[-1].str.split('.').str[0]
    index.name = 'id'
    sbu.set_index(index, inplace=True)
    
    already_existing = os.listdir(img_path)
    already_existing = [os.path.splitext(img)[0] for img in already_existing]
    print(f"Already existing: {len(already_existing)} from {len(index)} images")
    sbu = sbu[~sbu.index.isin(already_existing)]
    n_workers = os.cpu_count()*4
    with concurrent.futures.ThreadPoolExecutor(n_workers) as executor:
        futures = [executor.submit(fetch_single_image, url, img_path, idx) for idx, url in 
            tqdm(sbu['image_urls'].items(), total=len(sbu), desc="Scheduling tasks")]
        list(tqdm(concurrent.futures.as_completed(futures), total=len(sbu), desc="Downloading images"))

    n_failed = len(sbu) - len(os.listdir(img_path))
    print(f"Failed to download {n_failed} images (pairs). Percentage: {n_failed/len(sbu)*100:.2f}%")
    
if __name__ == "__main__":
    main()