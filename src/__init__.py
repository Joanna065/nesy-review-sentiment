from smartparams import Smart
from torch.optim import AdamW

from src.data.datasets.baseline_datasets import (
    IMDBBaselineDataset,
    MovieReviewsBaselineDataset,
    StanfordTreebankBaselineDataset,
    GoEmotionsBaselineDataset,
    AllegroReviewsBaselineDataset,
    Polemo2BaselineDataset,
    MultiemoBaselineDataset,
    GoEmotionsSentimentBaselineDataset,
)
from src.data.datasets.hurtbert_embedding_datasets import (
    IMDBHurtBertEmbeddingDataset,
    MovieReviewsHurtBertEmbeddingDataset,
    StanfordTreebankHurtBertEmbeddingDataset,
    GoEmotionsHurtBertEmbeddingDataset,
    AllegroReviewsHurtBertEmbeddingDataset,
    Polemo2HurtBertEmbeddingDataset,
    MultiemoHurtBertEmbeddingDataset,
)
from src.data.datasets.hurtbert_encoding_datasets import (
    MovieReviewsHurtBertEncodingDataset,
    IMDBHurtBertEncodingDataset,
    StanfordTreebankHurtBertEncodingDataset,
    GoEmotionsHurtBertEncodingDataset,
    AllegroReviewsHurtBertEncodingDataset,
    Polemo2HurtBertEncodingDataset,
    MultiemoHurtBertEncodingDataset,
)
from src.data.datasets.kepler_datasets import (
    IMDBKeplerDataset,
    MovieReviewsKeplerDataset,
    StanfordTreebankKeplerDataset,
    AllegroReviewsKeplerDataset,
    GoEmotionsKeplerDataset,
    Polemo2KeplerDataset,
    MultiemoKeplerDataset,
)
from src.data.datasets.senti_lare_datasets import (
    IMDBSentiLAREDataset,
    MovieReviewsSentiLAREDataset,
    StanfordTreebankSentiLAREDataset,
    GoEmotionsSentiLAREDataset,
    AllegroReviewsSentiLAREDataset,
    Polemo2SentiLAREDataset,
    MultiemoSentiLAREDataset,
    GoEmotionsSentimentSentiLAREDataset,
)

Smart.debug = True
Smart.register(
    IMDBBaselineDataset,
    MovieReviewsBaselineDataset,
    StanfordTreebankBaselineDataset,
    GoEmotionsBaselineDataset,
    AllegroReviewsBaselineDataset,
    Polemo2BaselineDataset,
    MultiemoBaselineDataset,
    GoEmotionsSentimentBaselineDataset,
    option='type',
)
Smart.register(
    IMDBHurtBertEncodingDataset,
    MovieReviewsHurtBertEncodingDataset,
    StanfordTreebankHurtBertEncodingDataset,
    GoEmotionsHurtBertEncodingDataset,
    AllegroReviewsHurtBertEncodingDataset,
    Polemo2HurtBertEncodingDataset,
    MultiemoHurtBertEncodingDataset,
    option='type',
)
Smart.register(
    IMDBHurtBertEmbeddingDataset,
    MovieReviewsHurtBertEmbeddingDataset,
    StanfordTreebankHurtBertEmbeddingDataset,
    GoEmotionsHurtBertEmbeddingDataset,
    AllegroReviewsHurtBertEmbeddingDataset,
    Polemo2HurtBertEmbeddingDataset,
    MultiemoHurtBertEmbeddingDataset,
    option='type',
)
Smart.register(
    IMDBKeplerDataset,
    MovieReviewsKeplerDataset,
    StanfordTreebankKeplerDataset,
    GoEmotionsKeplerDataset,
    AllegroReviewsKeplerDataset,
    Polemo2KeplerDataset,
    MultiemoKeplerDataset,
    option='type',
)
Smart.register(
    IMDBSentiLAREDataset,
    MovieReviewsSentiLAREDataset,
    StanfordTreebankSentiLAREDataset,
    GoEmotionsSentiLAREDataset,
    AllegroReviewsSentiLAREDataset,
    Polemo2SentiLAREDataset,
    MultiemoSentiLAREDataset,
    GoEmotionsSentimentSentiLAREDataset,
    option='type',
)

Smart.register(
    AdamW,
    option='smart',
)
