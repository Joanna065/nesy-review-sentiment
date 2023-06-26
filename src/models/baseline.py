from smartparams import Smart
from torch import Tensor
from torch.optim import Optimizer
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.models.classifier import Classifier


class BaselineModel(Classifier):
    NAME = 'baseline'

    def __init__(
        self,
        name: str,
        num_labels: int,
        optimizer: Smart[Optimizer],
        multilabel: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(num_labels=num_labels, multilabel=multilabel)
        self.name = name
        self.optimizer = optimizer
        self._get_bert_model()

        if gradient_checkpointing:
            self.bert_model.gradient_checkpointing_enable()

        if self.multilabel:
            self.bert_model.config.problem_type = 'multi_label_classification'

    def _get_bert_model(self) -> None:
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.name,
            num_labels=self.num_labels,
        )

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

    def step(self, batch: dict[str, Tensor], step_type: str) -> dict[str, Tensor]:
        outputs = self(batch)

        loss = outputs['loss']
        if step_type != 'test':
            self.log(name=f'{step_type}/loss', value=loss, on_epoch=False, on_step=True)

        output = {
            'loss': loss if step_type == 'train' else loss.detach(),
            'logits': outputs['logits'].detach(),
            'labels': batch['labels'].detach(),
            'ids': batch['ids'].detach(),
        }
        return output

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.optimizer.pop('weight_decay'),
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        warmup_steps = self.optimizer.pop('warmup_steps')
        optimizer = self.optimizer(
            params=optimizer_grouped_parameters,
        )

        if warmup_steps > 0:
            # calculate total learning steps
            # call len(self.train_dataloader()) should be fixed in pytorch-lightning v1.6
            self._total_train_steps = (
                self.trainer.max_epochs
                * len(self.trainer._data_connector._train_dataloader_source.dataloader())
                * self.trainer.accumulate_grad_batches
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self._total_train_steps,
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                },
            }
        else:
            return {'optimizer': optimizer}
