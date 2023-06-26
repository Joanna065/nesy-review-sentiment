import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class DisambiguatorOnline:
    AMUSE_WSD_URL = 'http://nlp.uniroma1.it/amuse-wsd/api/model'
    HEADERS = {'Content-Type': 'application/json', 'accept': 'application/json'}
    SUPPORTED_LANGS = ['AR', 'DE', 'EN', 'ES', 'FR', 'IT', 'NL', 'PT', 'RU', 'ZH', 'PL']

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: str,
        lang: str,
        save_jsonl_path: Path,
        batch: int = 100,
        tmp_filename: str = '__tmp_fetched',
    ):
        if lang not in self.SUPPORTED_LANGS:
            raise ValueError(
                "Language not supported in online WSD api version. Supported: %s",
                self.SUPPORTED_LANGS,
            )
        self.lang = lang
        self.text_col = text_col
        self.id_col = id_col
        self.batch_size = batch
        self.df = df
        self.already_fetched_tmp_path = Path.cwd().joinpath(f'{tmp_filename}.txt')
        self.save_dir = save_jsonl_path.parent
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_jsonl_path = save_jsonl_path

    def fetch_results(self):
        self._get_records(self.df)

        batch_num = math.ceil(len(self.records) / self.batch_size)
        num_results = 0
        tqdm_bar = tqdm(total=batch_num, desc=f"Fetched {num_results} wsd results")

        for ids, data in self._batchify():
            response = self.post_request(data=data, ids=ids)
            if response:
                self._write(ids=ids, response=response)
                num_results += len(ids)

                tqdm_bar.update()
                tqdm_bar.set_description(f"Fetched {num_results} wsd results")

        self._check_tmp()

    def jsonlines_to_json(self):
        lines = self.save_jsonl_path.open(mode='r').readlines()
        records = [json.loads(line) for line in lines]
        wsd_dict = defaultdict(dict)

        for record in records:
            for key, value in record.items():
                wsd_dict[key] = value

        json_path = self.save_dir.joinpath(f'{self.save_jsonl_path.stem}.json')

        with json_path.open(mode='w') as f:
            json.dump(wsd_dict, f, ensure_ascii=False, indent=2)

        self.already_fetched_tmp_path.unlink(missing_ok=True)

    def post_request(self, data: List[Dict[str, Any]], ids: List[str]) -> Optional[List[Dict]]:
        assert len(data) == len(ids)

        r = requests.post(
            self.AMUSE_WSD_URL,
            data=json.dumps([obj for obj in data], ensure_ascii=False),
            headers=self.HEADERS,
        )
        if r.status_code == 200:
            response = r.json()
            if len(response) == len(data):
                return response
        else:
            logger.error("Unsuccessful status code: %d", r.status_code)
            logger.error(f"Missed indexes: {ids}")
            logger.error(r.text)

        return None

    def _write(self, ids: List[str], response: List[Dict[str, Any]]):
        wsd_dict = dict(zip(ids, response))
        results = [json.dumps({key: val}, ensure_ascii=False) for key, val in wsd_dict.items()]

        self.save_jsonl_path.open(mode='a').writelines('\n'.join(results) + '\n')
        self.already_fetched_tmp_path.open(mode='a').writelines('\n'.join(ids) + '\n')

    def _get_records(self, df: pd.DataFrame) -> None:
        id_list = df[self.id_col].values

        if self.already_fetched_tmp_path.exists():
            already_fetched_ids = set(self.already_fetched_tmp_path.read_text().strip().split('\n'))
            id_list = [idx for idx in id_list if idx not in already_fetched_ids]
            logger.info("Skipping %d dataset ids", len(df) - len(id_list))

        df = df[df[self.id_col].isin(id_list)]
        df = df[[self.text_col]]

        if self.text_col != 'text':
            df = df.rename(columns={self.text_col: 'text'})

        df['lang'] = self.lang

        records = df.to_json(orient='records')
        records = json.loads(records)

        self.id_list = id_list
        self.records = records

        logger.info("Data samples to fetch: %d", len(records))

    def _batchify(self) -> Generator[Tuple[List[str], List[Dict[str, Any]]], None, None]:
        for idx in range(0, len(self.records), self.batch_size):
            data = self.records[idx : idx + self.batch_size]
            ids = self.id_list[idx : idx + self.batch_size]

            yield ids, data

    def _check_tmp(self):
        if not self.already_fetched_tmp_path.exists():
            return

        already_fetched_ids = set(self.already_fetched_tmp_path.read_text().strip().split('\n'))
        if len(already_fetched_ids) == len(self.df):
            self.already_fetched_tmp_path.unlink()

            self.jsonlines_to_json()
            self.save_jsonl_path.unlink()
