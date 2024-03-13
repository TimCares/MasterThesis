import sys
sys.path.append('../../../')
from config import COCO_CAPTIONS_PATH
import json

def write():
    with open(COCO_CAPTIONS_PATH+'dataset_coco_karpathy.json') as f:
        meta = json.load(f)['images']

    train_f = open('captions_train.txt', 'w')
    val_f = open('captions_val.txt', 'w')
    test_f = open('captions_test.txt', 'w')
    
    for m in meta:
        if m['split'] == 'train':
            f = train_f
        elif m['split'] == 'val':
            f = val_f
        elif m['split'] == 'test':
            f = test_f
        else:
            raise ValueError('Invalid split')
        
        for s in m['sentences']:
            f.write(s['raw']+'\n')
        f.write('\n')

if __name__ == '__main__':
    write()