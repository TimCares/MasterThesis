import multiprocessing
import os
import json
from typing import Dict, Any, List
import torch
from functools import partial
from tqdm import tqdm

def process_one_batch(batch_meta:Dict[str, Any], batch_size:int, data_path_new:str) -> List[Dict[str, Any]]:
    data = torch.load(batch_meta['path'])
    if 'data_path' in data:
        key = 'data_path'
    elif 'text' in data:
        key = 'text'
    else:
        raise ValueError("Unknown key for data in batch")
    
    result = []

    for i in range(len(data[key]), batch_size):
        sub_batch = {
            key: data[key][i:i+batch_size].copy(),
            'target': data['target'][i:i+batch_size].copy(),
            'modes': data['modes'],
        }
        offset_idx = batch_meta["batch_idx"]*i
        filename = f'{batch_meta["batch_idx"]}_{offset_idx}-{offset_idx+batch_size}.pt'
        out_path = os.path.join(data_path_new, filename)
        torch.save(sub_batch, out_path)
        result.append({
            "path": out_path,
            "batch_idx": batch_meta["batch_idx"],
        })
    return result

def main(batch_size:int, data_path:str):
    with open(os.path.join(data_path, 'index.json'), 'r', encoding='utf-8') as f:
        index = json.load(f)

    data_path_new = data_path + f'_batch_{batch_size}'

    prev_batch_size = index['datamodule']['batch_size']

    if prev_batch_size == batch_size:
        return
    
    assert prev_batch_size % batch_size == 0, "New batch size must be a divisor of the old batch size"
    
    index['datamodule']['batch_size'] = batch_size

    processor = partial(process_one_batch, batch_size=batch_size, data_path_new=data_path_new)
    new_index = []
    with multiprocessing.Pool() as pool:
        for batch_result in tqdm(pool.imap(processor, index['index'])):
            new_index.extend(batch_result)

    index['index'] = new_index

    with open(os.path.join(data_path_new, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f)


if "__name__" == "__main__":
    main()