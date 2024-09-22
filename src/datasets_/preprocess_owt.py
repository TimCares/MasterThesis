import os
import argparse
import glob
import logging
import shutil
import tarfile
import lzma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/workspace')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    data_path = args.data_path
    split = args.split
    extracted_data_path = os.path.join(data_path, 'openwebtext')

    pattern = os.path.join(data_path, f'urlsf_subset*.tar')
    files = glob.glob(pattern)

    if len(files)==0:
        raise FileNotFoundError(f"No tar files found under: {data_path}")

    logger.info(f"Found {len(files)} tar files, inflating...")
    for file in files:
        with tarfile.open(file, 'r') as tar:
            tar.extractall(data_path)
    pattern = os.path.join(extracted_data_path, '*.xz')
    files = glob.glob(pattern)
    for file in files:
        with lzma.open(file, 'rb') as compressed_file:
            decompressed_data = compressed_file.read()
        txt_file = file.replace('.xz', '')
        with open(txt_file, 'wb') as output_file:
            output_file.write(decompressed_data)
        os.remove(file)
    
    logger.info("Inflated all tar files.")

    logger.info("Cleaning...")
    files = os.listdir(extracted_data_path)
    for file in files:
        try:
            with open(os.path.join(extracted_data_path, file), 'r+', encoding='utf-8') as reader:
                lines = reader.readlines()
                reader.seek(0)
                reader.truncate()
                for line in lines:
                    line = line.strip()
                    if '\x00' not in line and line != '' and line != '---': # remove null bytes and empty lines
                        reader.write(line)
                        reader.write('\n')
        except:
            logger.error(f"Error cleaning {file}")
            exit(1)

    text_file = os.path.join(data_path, f'openwebtext.txt.{split}')
    logger.info("Joining...")
    with open(text_file, 'w', encoding='utf-8') as f:
        for file in files:
            path_to_file = os.path.join(extracted_data_path, file)
            with open(path_to_file, 'r', encoding='utf-8') as f2:
                f.write(f2.read())
            os.remove(path_to_file)
    shutil.rmtree(extracted_data_path)

    logger.info("Done!")

if __name__ == '__main__':
    main()