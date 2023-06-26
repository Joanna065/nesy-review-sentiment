import ast
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.settings import SENTIMENT_RES_DIR

AVAILABLE_ANNOTATIONS = [
    'sentiwordnetScore',
    'plwnSynsetId',
    'plwnSentimentScore',
    'plwnEmotionNames',
    'plwnEmotionValuations',
]

EMONAMES_FIXER_MAP = {
    'cieszenie sie': 'cieszenie się',
    'cieszenie się na': 'cieszenie się',
    'cieszenie się na coś oczekiwanego': 'cieszenie się',
    'smitek': 'smutek',
    'smute': 'smutek',
    'wstret': 'wstręt',
    'zaskoczenie czymś nieprzewidywanym': 'zaskoczenie',
    'złosć': 'złość',
    'dobro': 'dobro drugiego człowieka',
}

EMOVALUATIONS_FIXER_MAP = {
    ' dobro': 'dobro drugiego człowieka',
    'dobro': 'dobro drugiego człowieka',
    'dobro drugiego': 'dobro drugiego człowieka',
    ' nieużyteczność': 'nieużyteczność',
    'piekno': 'piękno',
    'szczęśćie': 'szczęście',
    'prwda': 'prawda',
}


class SentEmoAnnotatorAmuseOutput:
    def __init__(
        self,
        filepath: Path,
        save_path: Path,
    ):
        self.filepath = filepath
        self.save_path = save_path

        self.sentiwordnet_labels = set()
        self.sentiplwn_labels = set()
        self.plwn_emonames_labels = set()
        self.plwn_emovaluations_labels = set()

        self.pwn_sentiwordnet_mapping = self.get_sentiwordnet_scores()
        self.pwn_plwn_mapping = self.get_pwn_plwn_mapping()
        (
            self.plwn_synonymy_scores,
            self.plwn_hyponymy_scores,
            self.plwn_hypernymy_scores,
        ) = self.get_plwn_sentiment_scores()

        self.hyper_emonames, self.syn_emonames, self.hypo_emonames = self.get_pwn_emonames_map()
        (
            self.hyper_emovaluations,
            self.syn_emovaluations,
            self.hypo_emovaluations,
        ) = self.get_pwn_emovaluations_map()

    def process_document(self) -> None:
        with self.filepath.open(mode='r') as f:
            data = json.load(f)

        for sample_id in tqdm(data.keys(), desc="Processing annotations..."):
            tokens = data[sample_id]['tokens']
            updated_tokens = []
            for token in tokens:
                wn_offset = token['wnSynsetOffset']
                if wn_offset != 'O':
                    # assign SentiWordnet sentiment score
                    sentiment_score = self.pwn_sentiwordnet_mapping[wn_offset]
                    self.sentiwordnet_labels.add(sentiment_score)
                    token['sentiwordnetScore'] = sentiment_score

                    # assign polish Wordnet synset id
                    if wn_offset in self.pwn_plwn_mapping:
                        plwn_synset = str(self.pwn_plwn_mapping[wn_offset])
                    else:
                        plwn_synset = 'O'
                    token['plwnSynsetId'] = plwn_synset

                    # assign polish Wordnet sentiment scores,
                    # general score is hierarchically taken from synonym -> hyperonym -> hyponym
                    if wn_offset in self.plwn_synonymy_scores:
                        synonymy_score = self.plwn_synonymy_scores[wn_offset]
                        token['plwnSentimentScoreSynonymy'] = synonymy_score
                        token['plwnSentimentScore'] = synonymy_score

                    if wn_offset in self.plwn_hypernymy_scores:
                        hypernymy_score = self.plwn_hypernymy_scores[wn_offset]
                        token['plwnSentimentScoreHypernymy'] = hypernymy_score
                        if 'plwnSentimentScore' not in token:
                            token['plwnSentimentScore'] = hypernymy_score

                    if wn_offset in self.plwn_hyponymy_scores:
                        hyponymy_score = self.plwn_hyponymy_scores[wn_offset]
                        token['plwnSentimentScoreHyponymy'] = hyponymy_score
                        if 'plwnSentimentScore' not in token:
                            token['plwnSentimentScore'] = hyponymy_score

                    if 'plwnSentimentScore' in token:
                        self.sentiplwn_labels.add(token['plwnSentimentScore'])

                    # assign polish Wordnet emo annotations
                    if wn_offset in self.syn_emonames:
                        token['plwnEmotionNames'] = self._fix_emonames(self.syn_emonames[wn_offset])
                    elif wn_offset in self.hyper_emonames:
                        token['plwnEmotionNames'] = self._fix_emonames(
                            self.hyper_emonames[wn_offset]
                        )
                    elif wn_offset in self.hypo_emonames:
                        token['plwnEmotionNames'] = self._fix_emonames(
                            self.hypo_emonames[wn_offset]
                        )

                    if 'plwnEmotionNames' in token:
                        self.plwn_emonames_labels.update(token['plwnEmotionNames'])

                    if wn_offset in self.syn_emovaluations:
                        token['plwnEmotionValuations'] = self._fix_emovaluations(
                            self.syn_emovaluations[wn_offset]
                        )
                    elif wn_offset in self.hyper_emovaluations:
                        token['plwnEmotionValuations'] = self._fix_emovaluations(
                            self.hyper_emovaluations[wn_offset]
                        )
                    elif wn_offset in self.hypo_emovaluations:
                        token['plwnEmotionValuations'] = self._fix_emovaluations(
                            self.hypo_emovaluations[wn_offset]
                        )

                    if 'plwnEmotionValuations' in token:
                        self.plwn_emovaluations_labels.update(token['plwnEmotionValuations'])

                updated_tokens.append(token)
            data[sample_id]['tokens'] = updated_tokens

        with self.save_path.open(mode='w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _fix_emonames(emonames: list[str]) -> list[str]:
        return [EMONAMES_FIXER_MAP.get(emoname, emoname) for emoname in emonames]

    @staticmethod
    def _fix_emovaluations(emovaluations: list[str]) -> list[str]:
        return [EMOVALUATIONS_FIXER_MAP.get(emoval, emoval) for emoval in emovaluations]

    @staticmethod
    def get_sentiwordnet_scores() -> dict[str, float]:
        def _get_sentiment_score(pos: float, neg: float) -> float:
            return pos - neg

        def _get_wn_offset(wn_id: str, pos: str) -> str:
            wn_id = int(wn_id.lstrip('0'))
            return f'{wn_id}{pos}'

        SENTIWORDNET_PATH = SENTIMENT_RES_DIR.joinpath('SentiWordNet_3.0.0.txt')
        HEADERS = ['pos', 'id', 'pos_score', 'neg_score', 'synset_terms', 'gloss']

        with SENTIWORDNET_PATH.open(mode='r') as f:
            sentiwordnet_lines = f.readlines()

        df_sentiwordnet = pd.DataFrame(
            columns=HEADERS,
            data=[row.rstrip('\n').split('\t') for row in sentiwordnet_lines[26:117685]],
        )
        df_sentiwordnet = df_sentiwordnet.astype({'pos_score': float, 'neg_score': float})
        df_sentiwordnet['synset_offset'] = df_sentiwordnet[['id', 'pos']].apply(
            lambda x: _get_wn_offset(*x),
            axis=1,
        )
        df_sentiwordnet['sent_score'] = df_sentiwordnet[['pos_score', 'neg_score']].apply(
            lambda x: _get_sentiment_score(*x),
            axis=1,
        )
        pwn_sentiment_mapping = dict(
            zip(
                df_sentiwordnet['synset_offset'].values,
                df_sentiwordnet['sent_score'].values,
            )
        )
        return pwn_sentiment_mapping

    @staticmethod
    def get_pwn_plwn_mapping() -> dict:
        MAPPING_PWN_PLWN_PATH = SENTIMENT_RES_DIR.joinpath('plwn_pwn_mappings', 'pwn30-plwn32.txt')
        map_lines = MAPPING_PWN_PLWN_PATH.open(mode='r').readlines()
        pwn_plwn_mapping = dict()
        for line in map_lines:
            pwn_syn, plwn_syn = line.split('\t')
            pwn_syn = ''.join(pwn_syn.lstrip("0").split('-'))
            pwn_plwn_mapping[pwn_syn] = int(plwn_syn)

        return pwn_plwn_mapping

    @staticmethod
    def get_plwn_sentiment_scores() -> tuple[dict, dict, dict]:
        def get_mapping_dict(lines: list[str]) -> dict:
            mapper = dict()
            for line in lines:
                pwn_syn, sentiment = line.split('\t')
                pwn_syn = ''.join(pwn_syn.lstrip("0").split('-'))
                mapper[pwn_syn] = float(sentiment)
            return mapper

        SYNONYMY_SCORES_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_synonymy_scores.txt'
        )
        synonymy_lines = SYNONYMY_SCORES_PATH.open(mode='r').readlines()
        synonymy_scores = get_mapping_dict(lines=synonymy_lines)

        HYPONYMY_SCORES_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_hyponymy_scores.txt'
        )
        hyponymy_lines = HYPONYMY_SCORES_PATH.open(mode='r').readlines()
        hyponymy_scores = get_mapping_dict(lines=hyponymy_lines)

        HYPERNYMY_SCORES_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_hypernymy_scores.txt'
        )
        hypernymy_lines = HYPERNYMY_SCORES_PATH.open(mode='r').readlines()
        hypernymy_scores = get_mapping_dict(lines=hypernymy_lines)

        return synonymy_scores, hyponymy_scores, hypernymy_scores

    @staticmethod
    def _get_mapping_emo_dict(lines: list[str]) -> dict:
        mapper = dict()
        for line in lines:
            pwn_syn, emo_values = line.split('\t')
            pwn_syn = ''.join(pwn_syn.lstrip("0").split('-'))
            emo_values = emo_values.rstrip('\n')
            mapper[pwn_syn] = ast.literal_eval(emo_values)
        return mapper

    def get_pwn_emonames_map(self) -> tuple[dict, dict, dict]:
        HYPERNYMY_EMONAMES_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_hypernymy_emonames.txt'
        )
        hyper_emonames_lines = HYPERNYMY_EMONAMES_PATH.open(mode='r').readlines()
        hyper_emonames = self._get_mapping_emo_dict(lines=hyper_emonames_lines)

        SYNONYMY_EMONAMES_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_synonymy_emonames.txt'
        )
        syn_emonames_lines = SYNONYMY_EMONAMES_PATH.open(mode='r').readlines()
        syn_emonames = self._get_mapping_emo_dict(lines=syn_emonames_lines)

        HYPONYMY_EMONAMES_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_hyponymy_emonames.txt'
        )
        hypo_emonames_lines = HYPONYMY_EMONAMES_PATH.open(mode='r').readlines()
        hypo_emonames = self._get_mapping_emo_dict(lines=hypo_emonames_lines)

        return hyper_emonames, syn_emonames, hypo_emonames

    def get_pwn_emovaluations_map(self) -> tuple[dict, dict, dict]:
        HYPERNYMY_EMOVALUATIONS_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_hypernymy_emovaluations.txt'
        )
        hyper_emovaluations_lines = HYPERNYMY_EMOVALUATIONS_PATH.open(mode='r').readlines()
        hyper_emovaluations = self._get_mapping_emo_dict(lines=hyper_emovaluations_lines)

        SYNONYMY_EMOVALUATIONS_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_synonymy_emovaluations.txt'
        )
        syn_emovaluations_lines = SYNONYMY_EMOVALUATIONS_PATH.open(mode='r').readlines()
        syn_emovaluations = self._get_mapping_emo_dict(lines=syn_emovaluations_lines)

        HYPONYMY_EMOVALUATIONS_PATH = SENTIMENT_RES_DIR.joinpath(
            'plwn_pwn_mappings', 'pwn-plwn_hyponymy_emovaluations.txt'
        )
        hypo_emovaluations_lines = HYPONYMY_EMOVALUATIONS_PATH.open(mode='r').readlines()
        hypo_emovaluations = self._get_mapping_emo_dict(lines=hypo_emovaluations_lines)

        return hyper_emovaluations, syn_emovaluations, hypo_emovaluations
