import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.settings import DATA_DIR

SENTIMENT_SCORES = [
    -0.8,
    -0.4,
    0.0,
    0.4,
    0.8,
]
# 8 basic emotions
EMOTION_NAMES = [
    'radość',  # joy (Ekhman)
    'strach',  # fear (Ekhman)
    'zaskoczenie',  # surprise (Ekhman)
    'smutek',  # sadness (Ekhman)
    'wstręt',  # disgust (Ekhman)
    'złość',  # anger (Ekhman)
    'zaufanie',  # trust (Plutchik)
    'cieszenie się',  # anticipation (Plutchik)
]
# 12 fundamental human values postulated by Puzynina (Puzynina, 1992)
EMOTION_VALUATIONS = [
    'użyteczność',  # utility
    'dobro drugiego człowieka',  # another's good
    'prawda',  # truth
    'wiedza',  # knowledge
    'piękno',  # beauty
    'szczęście',  # happiness
    'nieużyteczność',  # futility
    'krzywda',  # harm
    'niewiedza',  # ignorance
    'błąd',  # error
    'brzydota',  # ugliness
    'nieszczęście',  # misfortune
]
NUM_FEATURES = 24

GOEMO_WSD_DIR = DATA_DIR.joinpath('goemotions', 'wsd_amuse_EN')
IMDB_WSD_DIR = DATA_DIR.joinpath('imdb', 'wsd_amuse_EN')
MR_WSD_DIR = DATA_DIR.joinpath('movie_reviews', 'wsd_amuse_EN')
ST_WSD_DIR = DATA_DIR.joinpath('stanford_treebank', 'wsd_amuse_EN')
KLEJ_AR_WSD_DIR = DATA_DIR.joinpath('klej_ar', 'wsd_amuse_PL')
POLEMO_WSD_DIR = DATA_DIR.joinpath('polemo2', 'wsd_amuse_PL')

datasets_dirs = [
    GOEMO_WSD_DIR,
    IMDB_WSD_DIR,
    MR_WSD_DIR,
    ST_WSD_DIR,
    KLEJ_AR_WSD_DIR,
    POLEMO_WSD_DIR,
]

splits = ['train', 'dev', 'test']
use_sentiwordnet = False

# Iterate through all datasets to get PWN ids present there and construct one-hot embeddings ---------------------------
PWN_ID_FEATURES_DICT = defaultdict(list)

for dataset_dir in datasets_dirs:
    for split in splits:
        if dataset_dir.parent.stem == 'polemo2':
            json_path = dataset_dir.joinpath(f'all_text.{split}.amuse_wsd.json')
        else:
            json_path = dataset_dir.joinpath(f'{split}.amuse_wsd.json')

        with json_path.open(mode='r') as f:
            data = json.load(f)

        for sample_id in tqdm(
            data.keys(),
            desc="Getting pwn ids sentemo one-hot embedding",
            total=len(data.keys()),
        ):
            sample_data = data[sample_id]
            tokens = sample_data['tokens']

            for token in tokens:
                pwn_id = token['wnSynsetOffset']

                if pwn_id in PWN_ID_FEATURES_DICT:
                    continue

                if pwn_id != "O":
                    # create dict with features keys in specific order
                    encode_dict = dict()
                    for sent_val in SENTIMENT_SCORES:
                        encode_dict[sent_val] = 0
                    for emo_name in EMOTION_NAMES:
                        encode_dict[emo_name] = 0
                    for emo_val in EMOTION_VALUATIONS:
                        encode_dict[emo_val] = 0

                    # get feature info and make one-hot
                    sent_score = None
                    if 'plwnSentimentScore' in token:
                        sent_score = float(token['plwnSentimentScore'])
                        encode_dict[sent_score] = 1
                    if use_sentiwordnet and sent_score is None and 'sentiwordnetScore' in token:
                        sent_score = float(token['sentiwordnetScore'])

                        # threshold to plWN sentiment values
                        plwn_thresholds = np.array(SENTIMENT_SCORES)
                        plwn_sent_idx = np.argmin(np.abs(plwn_thresholds - sent_score))
                        sent_score = SENTIMENT_SCORES[plwn_sent_idx]
                        encode_dict[sent_score] = 1

                    # add emotion names & valuations
                    if 'plwnEmotionNames' in token:
                        emo_names = token['plwnEmotionNames']
                        for emo_name in emo_names:
                            encode_dict[emo_name] = 1

                    if 'plwnEmotionValuations' in token:
                        emo_valuations = token['plwnEmotionValuations']
                        for emo_val in emo_valuations:
                            encode_dict[emo_val] = 1

                    # delete neutral sentiment
                    encode_dict.pop(0.0)
                    features = list(encode_dict.values())
                    if any(features):
                        PWN_ID_FEATURES_DICT[pwn_id] = features

# Save file ------------------------------------------------------------------------------------------------------------
SAVE_FILEPATH = DATA_DIR.joinpath(f'hurtbert_embedding_sentiwordnet={use_sentiwordnet}.vec')

with SAVE_FILEPATH.open(mode='w') as f:
    # add pad token
    f.write(f'<pad>\t{" ".join([str(0) for _ in range(NUM_FEATURES)])}\n')
    # add unk token
    f.write(f'<unk>\t{" ".join([str(0) for _ in range(NUM_FEATURES)])}\n')
    for key, value in PWN_ID_FEATURES_DICT.items():
        one_hot_str = ' '.join([str(x) for x in value])
        f.write(f'{key}\t{one_hot_str}\n')
