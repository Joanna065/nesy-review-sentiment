from src.data_preprocessing.annotator_sent_emo import SentEmoAnnotatorAmuseOutput
from src.settings import DATA_DIR

sentiwordnet_labels = set()
sentiplwn_labels = set()
emonames_labels = set()
emovaluations_labels = set()

dataset = 'multiemo'
language = 'EN'

for split in ['train', 'dev', 'test']:
    print(f'Split name: {split}')
    DATASET_DIR = DATA_DIR.joinpath(dataset, f'wsd_amuse_{language}')
    WSD_FILEPATH = DATASET_DIR.joinpath(f'{split}.amuse_wsd.json')
    SAVE_WSD_FILEPATH = DATASET_DIR.joinpath(f'{split}.amuse_wsd.json')

    amuse_annotator = SentEmoAnnotatorAmuseOutput(
        filepath=WSD_FILEPATH,
        save_path=SAVE_WSD_FILEPATH,
    )
    amuse_annotator.process_document()
    sentiwordnet_labels.update(amuse_annotator.sentiwordnet_labels)
    sentiplwn_labels.update(amuse_annotator.sentiplwn_labels)
    emonames_labels.update(amuse_annotator.plwn_emonames_labels)
    emovaluations_labels.update(amuse_annotator.plwn_emovaluations_labels)

### save annotation LABELS possible values
with SAVE_WSD_FILEPATH.parent.joinpath('sentiwordnet_scores.txt').open(mode='w') as f:
    f.writelines(sorted([f'{x}\n' for x in sentiwordnet_labels]))

with SAVE_WSD_FILEPATH.parent.joinpath('sentiplwn_scores.txt').open(mode='w') as f:
    f.writelines(sorted([f'{x}\n' for x in sentiplwn_labels]))

with SAVE_WSD_FILEPATH.parent.joinpath('emoplwn_names.txt').open(mode='w') as f:
    f.writelines(sorted([f'{x}\n' for x in emonames_labels]))

with SAVE_WSD_FILEPATH.parent.joinpath('emoplwn_valuations.txt').open(mode='w') as f:
    f.writelines(sorted([f'{x}\n' for x in emovaluations_labels]))
