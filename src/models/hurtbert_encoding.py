import torch
from smartparams import Smart
from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import AutoModel

from src.models.baseline import BaselineModel


class HurtbertEncodingModel(BaselineModel):
    NAME = 'hurtbert_encoding'

    def __init__(
        self,
        name: str,
        num_labels: int,
        optimizer: Smart[Optimizer],
        multilabel: bool = False,
        gradient_checkpointing: bool = False,
        bert_dense_dim: int = 256,
        sentiemo_features_size: int = 24,
    ) -> None:
        super().__init__(
            name=name,
            num_labels=num_labels,
            optimizer=optimizer,
            multilabel=multilabel,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.bert_dense_layer = nn.Sequential(
            nn.Linear(
                in_features=self.bert_model.config.hidden_size,
                out_features=bert_dense_dim,
            ),
            nn.ReLU(),
        )
        self.sentiemo_features_size = sentiemo_features_size
        self.classifier_layer = nn.Linear(
            in_features=bert_dense_dim + self.sentiemo_features_size,
            out_features=num_labels,
        )
        if self.multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def _get_bert_model(self) -> None:
        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.name,
        )

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        bert_pooler_output = self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        ).pooler_output
        text_embeddings = self.bert_dense_layer(bert_pooler_output)
        hurtbert_encoding = batch['hurtbert_encodings']

        concat_in = torch.cat((text_embeddings, hurtbert_encoding), dim=1).float()
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
