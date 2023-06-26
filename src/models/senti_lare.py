from pytorch_transformers import RobertaConfig
from smartparams import Smart
from torch import Tensor
from torch.optim import Optimizer

from src.models.baseline import BaselineModel
from src.models.modules.modeling_sentilare_roberta import (
    RobertaForSequenceClassification,
    RobertaForMultiLabelClassification,
)
from src.settings import SENTI_LARE_DIR

# SentiLARE's input embeddings include POS embedding, word-level sentiment polarity embedding,
# and sentence-level sentiment polarity embedding (which is set to be unknown during fine-tuning).
is_pos_embedding = True
is_senti_embedding = True
is_polarity_embedding = True


class SentiLAREModel(BaselineModel):
    NAME = 'sentiLARE'
    CHECKPOINT_PATH = SENTI_LARE_DIR.joinpath('sentiLARE_pretrained')

    def __init__(
        self,
        num_labels: int,
        optimizer: Smart[Optimizer],
        name: str = 'roberta',
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

    def _get_bert_model(self) -> None:
        config = RobertaConfig.from_pretrained(str(self.CHECKPOINT_PATH))
        config.num_labels = self.num_labels

        if self.multilabel:
            self.bert_model = RobertaForMultiLabelClassification.from_pretrained(
                str(self.CHECKPOINT_PATH),
                from_tf=bool('.ckpt' in str(self.CHECKPOINT_PATH)),
                config=config,
                pos_tag_embedding=is_pos_embedding,
                senti_embedding=is_senti_embedding,
                polarity_embedding=is_polarity_embedding,
            )
        else:
            self.bert_model = RobertaForSequenceClassification.from_pretrained(
                str(self.CHECKPOINT_PATH),
                from_tf=bool('.ckpt' in str(self.CHECKPOINT_PATH)),
                config=config,
                pos_tag_embedding=is_pos_embedding,
                senti_embedding=is_senti_embedding,
                polarity_embedding=is_polarity_embedding,
            )

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return self.bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            pos_tag_ids=batch['pos_tag_ids'],
            senti_word_ids=batch['senti_word_ids'],
            polarity_ids=batch['polarity_ids'],
        )

    def step(self, batch: dict[str, Tensor], step_type: str) -> dict[str, Tensor]:
        outputs = self(batch)

        loss = outputs[0]  # pytorch_transformers model returns a tuple instead of dictionary
        if step_type != 'test':
            self.log(name=f'{step_type}/loss', value=loss, on_epoch=False, on_step=True)

        output = {
            'loss': loss if step_type == 'train' else loss.detach(),
            'logits': outputs[1].detach(),
            'labels': batch['labels'].detach(),
            'ids': batch['ids'].detach(),
        }
        return output
