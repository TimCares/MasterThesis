import sys
sys.path.append('../../../')
from config import COCO_CAPTIONS_PATH
import json
import os

def create_karpathy_split():
    with open(COCO_CAPTIONS_PATH+'dataset_coco.json') as f:
        data = json.load(f)

    for i in range(len(data['images'])):
        if data['images'][i]['split'] == 'restval':
            data['images'][i]['split'] = 'train'

    for i in range(len(data['images'])):
        data['images'][i]['full_path'] = os.path.join(data['images'][i]['filepath'], data['images'][i]['filename'])

    with open(COCO_CAPTIONS_PATH+'dataset_coco_karpathy.json', 'w') as f:
        json.dump(data, f)

    os.mkdir(COCO_CAPTIONS_PATH+'karpathy_train')
    os.mkdir(COCO_CAPTIONS_PATH+'karpathy_test')
    os.mkdir(COCO_CAPTIONS_PATH+'karpathy_val')

    for d in data['images']:
        os.symlink(COCO_CAPTIONS_PATH+d['full_path'], COCO_CAPTIONS_PATH+'karpathy_'+d['split']+'/'+d['filename'])

if __name__ == '__main__':
    create_karpathy_split()