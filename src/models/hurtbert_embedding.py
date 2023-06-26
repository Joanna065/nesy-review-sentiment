from typing import Optional

import torch
from smartparams import Smart
from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import AutoModel

from src.models.baseline import BaselineModel


class HurtbertEmbeddingModel(BaselineModel):
    NAME = 'hurtbert_embedding'

    def __init__(
        self,
        name: str,
        num_labels: int,
        optimizer: Smart[Optimizer],
        multilabel: bool = False,
        gradient_checkpointing: bool = False,
        bert_dense_dim: int = 256,
        lstm_hidden_size: int = 32,
        lstm_dense_dim: int = 16,
        sentiemo_features_size: int = 24,
    ) -> None:
        super().__init__(
            name=name,
            num_labels=num_labels,
            optimizer=optimizer,
            multilabel=multilabel,
            gradient_checkpointing=gradient_checkpointing,
        )
        if self.multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.bert_dense_layer = nn.Sequential(
            nn.Linear(
                in_features=self.bert_model.config.hidden_size,
                out_features=bert_dense_dim,
            ),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=sentiemo_features_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        self.lstm_dense_layer = nn.Sequential(
            nn.Linear(
                in_features=lstm_hidden_size,
                out_features=lstm_dense_dim,
            ),
            nn.ReLU(),
        )
        self.classifier_layer = nn.Linear(
            in_features=bert_dense_dim + lstm_dense_dim,
            out_features=num_labels,
        )

    def _get_bert_model(self) -> None:
        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.name,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        embedding_weights = self.trainer.datamodule.embedding_weights
        self.embedding_layer = nn.Embedding.from_pretrained(
            embedding_weights,
            freeze=True,
            padding_idx=0,
        )

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        # text model branch
        bert_pooler_output = self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        ).pooler_output
        text_embeddings = self.bert_dense_layer(bert_pooler_output)

        # neurosymbolic model branch
        embed_out = self.embedding_layer(batch['hurtbert_embeddings'].long())
        lstm_out, (hidden, cell) = self.lstm(embed_out)
        neurosymbolic_out = self.lstm_dense_layer(hidden.squeeze())

        # concat branches
        concat_in = torch.cat((text_embeddings, neurosymbolic_out), dim=1).float()
        logits = self.classifier_layer(concat_in)

        return logits

    def step(self, batch: dict[str, Tensor], step_type: str) -> dict[str, Tensor]:
        logits = self(batch)
        loss = self.loss_fn(logits, batch['labels'])

        if step_type != 'test':
            self.log(name=f'{step_type}/loss', value=loss, on_epoch=False, on_step=True)

        output = {
            'loss': loss if step_type == 'train' else loss.detach(),
            'logits': logits.detach(),
            'labels': batch['labels'].detach(),
            'ids': batch['ids'].detach(),
        }
        return output
