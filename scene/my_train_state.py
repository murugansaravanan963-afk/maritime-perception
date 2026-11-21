import json
from pathlib import Path

DATA = Path('D:/Projects/maritime-dataset/scene/data/scene_lars_v1')
STATE_KEYS = ['calm', 'fog', 'glitter', 'low_light']

meta = json.load(open(DATA / 'scene_labels.json'))
records = meta['records']
train = [r for r in records if r['split']== 'train']

print(len(records), len(train))
print('train[0][\'id\']', [train[0]['k'] for k in STATE_KEYS])
