import json
from collections import defaultdict
from itertools import product

import pandas as pd
from tqdm import tqdm

from src.settings import SENTIMENT_RES_DIR, DATA_DIR, KEPLER_KE_DIR

PLWN_DIR = SENTIMENT_RES_DIR.joinpath('plwordnet')

df_syn_relations = pd.read_csv(PLWN_DIR.joinpath('syn_rel.csv'), sep='\t')

# plWN - PWN mappings by relation --------------------------------------------------------------------------------------
df_plwn_pwn_mapping = pd.read_csv(
    SENTIMENT_RES_DIR.joinpath('plwn_pwn_mappings', 'mapping_plwn_i-links_i-all.txt'),
    sep='\t',
)
SYN_RELTYPES_PLWN_PWN = [
    'międzyjęzykowa_synonimia_częściowa_plWN-PWN',
    'międzyjęzykowa_synonimia_częściowa_PWN-plWN',
    'Syn_plWN-PWN',
    'Syn_PWN-plWN',
]

plwn_pl_en_map_dict = defaultdict(set)
for idx, row in df_plwn_pwn_mapping.iterrows():
    relation_name = row['name']

    # synonyms
    if relation_name in [
        'Syn_plWN-PWN',
        'międzyjęzykowa_synonimia_częściowa_plWN-PWN',
        'międzyjęzykowa_synonimia_międzyparadygmatyczna_made_of_plWN-PWN',
    ]:
        plwn_pl_parent_id = str(row['parent_id'])
        plwn_en_child_id = str(row['child_id'])
        plwn_pl_en_map_dict[plwn_pl_parent_id].add(plwn_en_child_id)

    if relation_name in [
        'Syn_PWN-plWN',
        'międzyjęzykowa_synonimia_częściowa_PWN-plWN',
        'międzyjęzykowa_synonimia_międzyparadygmatyczna_made_of_PWN-plWN',
    ]:
        plwn_en_parent_id = str(row['parent_id'])
        plwn_pl_child_id = str(row['child_id'])

        plwn_pl_en_map_dict[plwn_pl_child_id].add(plwn_en_parent_id)

# create synsets set which appeared in datasets ------------------------------------------------------------------------
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
PLWN_EN_SENSE_IDS = set()

for dataset_dir in datasets_dirs:
    print(f'Dataset: {dataset_dir.parent.stem}')
    for split in splits:
        if dataset_dir.parent.stem == 'polemo2':
            json_path = dataset_dir.joinpath(f'all_text.{split}.amuse_wsd.json')
        else:
            json_path = dataset_dir.joinpath(f'{split}.amuse_wsd.json')

        with json_path.open(mode='r') as f:
            data = json.load(f)

        for sample_id in tqdm(
            data.keys(),
            desc="Getting plWN sense ids appeared in datasets...",
            total=len(data.keys()),
        ):
            sample_data = data[sample_id]
            tokens = sample_data['tokens']

            for token in tokens:
                if 'plwnSynsetId' in token:
                    plwn_en_id = token['plwnSynsetId']
                    if plwn_en_id != "O":
                        PLWN_EN_SENSE_IDS.add(int(plwn_en_id))

# filter to those senses ids which are able to map to en-plWN ids and are present in datasets (to shorten learning KE)
# discard reltypes `plWN-PWN` and `PWN-plWN` ---------------------------------------------------------------------------
mapped_parents_list = []
mapped_childs_list = []
mapped_reltypes_list = []

for idx, row in tqdm(df_syn_relations.iterrows(), total=len(df_syn_relations)):
    parent = str(row['parent_id'])
    child = str(row['child_id'])
    reltype = row['reltype']

    if 'plWN-PWN' in reltype or 'PWN-plWN' in reltype:
        continue

    # check mapping existence
    if parent in plwn_pl_en_map_dict and child in plwn_pl_en_map_dict:
        mapped_parents = plwn_pl_en_map_dict[parent]
        mapped_childs = plwn_pl_en_map_dict[child]

        pairs = product(mapped_parents, mapped_childs)
        for parent_id, child_id in pairs:
            # check existence in datasets
            if int(parent_id) in PLWN_EN_SENSE_IDS and int(child_id) in PLWN_EN_SENSE_IDS:
                mapped_parents_list.append(parent_id)
                mapped_childs_list.append(child_id)
                mapped_reltypes_list.append(reltype)

df_syn_rel_mapped = pd.DataFrame(
    {
        'parent_id': mapped_parents_list,
        'child_id': mapped_childs_list,
        'reltype': mapped_reltypes_list,
    }
)
print(
    f'Filter to mapped to en-plWN ids and datasets existence ids - triplets len: {len(df_syn_rel_mapped)}'
)

# filter hiperonimia and hiponimia as symmetric relation ---------------------------------------------------------------
symmetric_relations_occur = set()

parents_list = []
childs_list = []
reltypes_list = []
for idx, row in tqdm(df_syn_rel_mapped.iterrows(), total=len(df_syn_rel_mapped)):
    parent = int(row['parent_id'])
    child = int(row['child_id'])
    reltype = row['reltype']

    if reltype == 'hiperonimia' or reltype == 'hiponimia':
        pair = tuple(sorted([parent, child]))
        if pair in symmetric_relations_occur:
            continue
        else:
            symmetric_relations_occur.add(pair)

    parents_list.append(parent)
    childs_list.append(child)
    reltypes_list.append(reltype)

df_syn_rel_mapped_filtered = pd.DataFrame(
    {
        'parent_id': parents_list,
        'child_id': childs_list,
        'reltype': reltypes_list,
    }
)
print(f'Filter symmetric relations - triplets len: {len(df_syn_rel_mapped_filtered)}')

# choose most common reltypes ------------------------------------------------------------------------------------------
reltype_counts = df_syn_rel_mapped_filtered.reltype.value_counts()

top_reltypes = reltype_counts[reltype_counts > 100]
CHOSEN_RELTYPES = top_reltypes.index.values.tolist()

df_syn_rel_mapped_filtered = df_syn_rel_mapped_filtered[
    df_syn_rel_mapped_filtered['reltype'].isin(CHOSEN_RELTYPES)
]
df_syn_rel_mapped_filtered.to_csv(KEPLER_KE_DIR.joinpath('triplets.csv'), sep='\t', index=False)
