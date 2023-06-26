from typing import Optional, Any

import torch
import torch.nn.functional as F
from smartparams import Smart
from torch import Tensor, nn
from torch.optim import Optimizer
from transformers import AutoModel

from src.models.baseline import BaselineModel


class KeplerModel(BaselineModel):
    NAME = 'kepler_tailored'

    def __init__(
        self,
        name: str,
        num_labels: int,
        optimizer: Smart[Optimizer],
        gamma: int = 1,
        multilabel: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            num_labels=num_labels,
            optimizer=optimizer,
            multilabel=multilabel,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.classifier = nn.Linear(
            in_features=self.bert_model.config.hidden_size,
            out_features=num_labels,
        )
        self.gamma = gamma

        if self.multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self.embedding = nn.Embedding(
            num_embeddings=self.trainer.datamodule.num_relations,
            embedding_dim=self.bert_model.config.hidden_size,
        )

    def _get_bert_model(self) -> None:
        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.name,
        )

    def bert_forward(self, batch):
        return self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if 'kepler' in batch:
            return self._forward_kepler_ke(batch)

        return self._forward_task(batch)

    def _forward_task(self, batch: dict[str, Any]) -> dict[str, Any]:
        pooler_output = self._forward_bert(batch)
        return {'logits': self.classifier(pooler_output)}

    def _forward_bert(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        ).pooler_output

    def _forward_kepler_ke(self, batch: dict[str, Any]) -> dict[str, Any]:
        # task
        task_outputs = self._forward_task(batch['task'])

        # knowledge plWN
        kepler_batch = batch['kepler']

        pos_heads_outputs = self._forward_bert(kepler_batch['positive_heads'])
        pos_rels_outputs = self.embedding(kepler_batch['positive_relations'])
        pos_tails_outputs = self._forward_bert(kepler_batch['positive_tails'])
        neg_heads_outputs = [
            self._forward_bert(neg_head) for neg_head in kepler_batch['negative_heads']
        ]
        neg_rels_outputs = [
            self.embedding(neg_rel) for neg_rel in kepler_batch['negative_relations']
        ]
        neg_tails_outputs = [
            self._forward_bert(neg_tail) for neg_tail in kepler_batch['negative_tails']
        ]

        return dict(
            task_logits=task_outputs['logits'],
            pos_head_outputs=pos_heads_outputs,
            pos_rels_outputs=pos_rels_outputs,
            pos_tails_outputs=pos_tails_outputs,
            neg_heads_outputs=neg_heads_outputs,
            neg_rels_outputs=neg_rels_outputs,
            neg_tails_outputs=neg_tails_outputs,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        outputs = self(batch)

        # task loss (nlp)
        task_logits = outputs['task_logits']
        labels = batch['task']['labels']
        task_loss = self.loss_fn(task_logits, labels)
        self.log(name=f'train/task_loss', value=task_loss, on_epoch=False, on_step=True)

        # KE loss (knowledge graph)
        ke_loss = self.ke_loss(outputs)
        self.log(name=f'train/ke_loss', value=ke_loss, on_epoch=False, on_step=True)

        loss = task_loss + ke_loss
        self.log(name=f'train/loss', value=loss, on_epoch=False, on_step=True)
        return {
            'loss': loss,
            'logits': task_logits.detach(),
            'labels': labels.detach(),
            'ids': batch['task']['ids'].detach(),
        }

    def validation_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ):
        return self._eval_step(batch, step_type='val')

    def test_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ):
        return self._eval_step(batch, step_type='test')

    def _eval_step(
        self,
        batch: dict[str, Tensor],
        step_type: str,
    ) -> dict[str, Tensor]:
        assert step_type in ['val', 'test']

        outputs = self(batch)
        logits = outputs['logits']
        labels = batch['labels']

        loss = self.loss_fn(logits, labels)
        self.log(name=f'{step_type}/loss', value=loss, on_epoch=False, on_step=True)

        return {
            'loss': loss.detach(),
            'logits': logits.detach(),
            'labels': labels.detach(),
            'ids': batch['ids'].detach(),
        }

    def ke_loss(self, outputs: dict[str, Any]):
        pos_head_outputs = outputs['pos_head_outputs']
        pos_rels_outputs = outputs['pos_rels_outputs']
        pos_tails_outputs = outputs['pos_tails_outputs']
        neg_heads_outputs = outputs['neg_heads_outputs']
        neg_rels_outputs = outputs['neg_rels_outputs']
        neg_tails_outputs = outputs['neg_tails_outputs']

        positive_score = self.calc_trans_e(
            head=pos_head_outputs,
            relation=pos_rels_outputs,
            tail=pos_tails_outputs,
        ).unsqueeze(1)

        negative_scores = [
            self.calc_trans_e(
                head=neg_head,
                relation=neg_rel,
                tail=neg_tail,
            ).unsqueeze(1)
            for neg_head, neg_rel, neg_tail in zip(
                neg_heads_outputs, neg_rels_outputs, neg_tails_outputs
            )
        ]
        negative_score = torch.torch.cat(negative_scores, dim=1)

        positive_loss = F.logsigmoid(positive_score).squeeze(dim=1)
        negative_loss = F.logsigmoid(-negative_score).mean(dim=1)
        loss = (-positive_loss.mean() - negative_loss.mean()) / 2.0

        return loss

    def calc_trans_e(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma - torch.norm(score, p=2, dim=1)
        return score
