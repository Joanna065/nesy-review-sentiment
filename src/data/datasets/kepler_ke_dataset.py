import logging
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import Callable, Any

import pandas as pd
from torch.utils.data.dataset import Dataset

from src.settings import KEPLER_KE_DIR

logger = logging.getLogger(__name__)


class KEDataset(Dataset):
    def __init__(
        self,
        train_sense_ids_set: set[int],
        processor: Callable,
        path_triplets: Path = KEPLER_KE_DIR.joinpath('triplets.csv'),
        path_definitions: Path = KEPLER_KE_DIR.joinpath('plwn_senses_comments.csv'),
    ) -> None:
        super().__init__()
        self._train_sense_ids_set = train_sense_ids_set
        logger.info(f"Kepler knowledge for plWN sense ids num: {len(self._train_sense_ids_set)}")
        self._processor = processor

        df_triplets = pd.read_csv(
            path_triplets,
            sep='\t',
            dtype={'parent_id': int, 'child_id': int},
        )
        df_sense_definitions = pd.read_csv(
            path_definitions,
            sep='\t',
            dtype={'plwn_id': int},
        )

        # trim triplets depending on train ids set
        df_filtered_triplets = self._filter_sense_ids_by_train(
            df_triplets=df_triplets,
            train_sense_ids_set=self._train_sense_ids_set,
        )

        self.relation_set = list(sorted(set(df_filtered_triplets['reltype'].values.tolist())))
        self._get_positive_negative_samples(
            df_triplets=df_filtered_triplets,
            df_sense_definitions=df_sense_definitions,
        )
        assert len(self.pos_rels) == len(self.neg_rels)
        assert len(self.pos_rels) == len(self.pos_tails)
        assert len(self.pos_rels) == len(self.pos_heads)
        assert len(self.neg_heads) == len(self.neg_rels)
        assert len(self.neg_tails) == len(self.neg_rels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = index % len(self.pos_rels)
        return dict(
            positive_head=self._processor(self.pos_heads[index]),
            positive_relation=self.pos_rels[index],
            positive_tail=self._processor(self.pos_tails[index]),
            negative_heads=[self._processor(head) for head in self.neg_heads[index]],
            negative_relations=self.neg_rels[index],
            negative_tails=[self._processor(tail) for tail in self.neg_tails[index]],
        )

    def __len__(self):
        return 1_000_000

    def _get_positive_negative_samples(
        self,
        df_triplets: pd.DataFrame,
        df_sense_definitions: pd.DataFrame,
    ) -> None:
        sense_to_text_map = dict(zip(df_sense_definitions.plwn_id, df_sense_definitions.comment))

        all_heads = df_triplets['parent_id'].values.tolist()
        all_tails = df_triplets['child_id'].values.tolist()

        self.pos_heads = [sense_to_text_map[head] for head in all_heads]
        self.pos_rels = [self.relation_set.index(rel) for rel in df_triplets.reltype]
        self.pos_tails = [sense_to_text_map[tail] for tail in all_tails]

        # --- get negative samples ---
        positive_samples = list(
            zip(df_triplets.parent_id, df_triplets.reltype, df_triplets.child_id)
        )

        pos_tails = defaultdict(set)
        pos_heads = defaultdict(set)
        for head, rel, tail in positive_samples:
            pos_tails[head, rel].add(tail)
            pos_heads[tail, rel].add(head)

        all_heads = set(all_heads)
        all_tails = set(all_tails)
        self.neg_heads = []
        self.neg_rels = []
        self.neg_tails = []
        for head, rel, tail in positive_samples:
            neg_tails = all_tails.difference(pos_tails[head, rel])
            neg_heads = all_heads.difference(pos_heads[tail, rel])
            neg_tail = sample(neg_tails, k=1)[0]
            neg_head = sample(neg_heads, k=1)[0]

            self.neg_heads.append([sense_to_text_map[head], sense_to_text_map[neg_head]])
            self.neg_rels.append([self.relation_set.index(rel)] * 2)
            self.neg_tails.append([sense_to_text_map[neg_tail], sense_to_text_map[tail]])

    @staticmethod
    def _filter_sense_ids_by_train(
        df_triplets: pd.DataFrame,
        train_sense_ids_set: set[int],
    ) -> pd.DataFrame:
        selected_indices = []
        for idx, row in df_triplets.iterrows():
            parent_id = row['parent_id']
            child_id = row['child_id']

            if parent_id in train_sense_ids_set and child_id in train_sense_ids_set:
                selected_indices.append(idx)

        logger.info(f"Selected {len(selected_indices)} triplet rows for train set.")

        df_filtered_triplets = df_triplets.iloc[selected_indices].copy(deep=True)
        df_filtered_triplets = df_filtered_triplets.reset_index()
        return df_filtered_triplets
