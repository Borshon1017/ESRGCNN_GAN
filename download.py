import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")

def download_and_extract_div2k(destination):
    urls = {
        'DIV2K_train_HR.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'DIV2K_train_LR_bicubic_X2.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
        'DIV2K_train_LR_bicubic_X3.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip',
        'DIV2K_train_LR_bicubic_X4.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip'
    }

    os.makedirs(destination, exist_ok=True)

    for filename, url in urls.items():
        print(f"Downloading {filename}...")
        download_file(url, os.path.join(destination, filename))
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(os.path.join(destination, filename), 'r') as zip_ref:
            zip_ref.extractall(destination)
        os.remove(os.path.join(destination, filename))
        print(f"Finished {filename}")

if __name__ == "__main__":
    download_and_extract_div2k('dataset/DIV2K')
