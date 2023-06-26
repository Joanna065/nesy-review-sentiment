# A review of neurosymbolics method for text classification

Master Thesis university project about NeSy method review for sentiment classification task.

## Polish datasets
### Allegro review
Allegro Reviews is a __sentiment__ analysis dataset, 
consisting of 11,588 product reviews written in Polish 
and extracted from Allegro.pl - a popular e-commerce marketplace. 
Each review contains at least 50 words and has a rating on a scale 
from one (negative review) to five (positive review).

Train / dev / test split are already provided. 
Can be downloaded via huggingface datasets.

### PolEmo2.0
The PolEmo2.0 is a set of online reviews from four domains: medicine, products,
reviews (university) and hotels. There are txt files available with manual annotation
for whole texts and selected sentences.
The task is to predict the __sentiment__ of a review.
Train / dev / test split are already provided. 

### Emotions PL (from Polemo2.0)
PolEmo2.0 annotated with emotional dimensions on text and sentence level.
Each example was annotated by several linguists.

## English datasets
The first three datasets are also provided in SentiLARE format. 
See [paper](https://arxiv.org/abs/1911.02493) 
and [code](https://github.com/thu-coai/SentiLARE) - from there the splits were
downloaded and used in this work.

### IMDB
Firstly introduced in [paper](https://aclanthology.org/P11-1015/),
the IMDb Movie Reviews dataset is a __binary__ sentiment analysis dataset
consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled 
as positive or negative. 
Contains an even number of positive and negative reviews. Only highly 
polarizing reviews are considered. A negative review has a score ≤ 4 out of 10,
and a positive review has a score ≥ 7 out of 10. No more than 30 reviews are 
included per movie.

Dataset cleaned from `'<br /><br />'` texts in order to properly do WSD.

### Movie Reviews
Dataset introduced in [paper](https://aclanthology.org/P05-1015/) include
collections of movie-review documents labeled with respect to their overall 
sentiment polarity (positive or negative) or subjective rating 
(e.g., "two and a half stars") and sentences labeled with respect to their 
subjectivity status (subjective or objective) or polarity. 
Raw data available at [link](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

### Stanford Treebank
Presented in [paper](https://aclanthology.org/D13-1170/)
is a corpus with fully labeled parse trees that allows for a complete analysis 
of the compositional effects of sentiment in language. The corpus is based on
the dataset introduced by Pang and Lee (2005) and consists of 11,855 single
sentences extracted from movie reviews. It was parsed with the Stanford parser
and includes a total of 215,154 unique phrases from those parse trees, each 
annotated by 3 human judges.

Each phrase is labelled as either negative, somewhat negative, neutral, somewhat
positive or positive. The corpus with all 5 labels is referred to as __SST-5__ or _SST
fine-grained_. Binary classification experiments on full sentences 
(negative or somewhat negative vs somewhat positive or positive with neutral 
sentences discarded) refer to the dataset as __SST-2__ or _SST binary_.

### GoEmotions
The [GoEmotions](https://arxiv.org/pdf/2005.00547.pdf) dataset contains 
58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral.
This dataset is intended for _multi-class_, _multi-label_ emotion classification.
Train / dev / test split are already provided. They include examples 
where there is agreement between at least 2 raters. 
- 43 410 training examples (train.tsv)
- 5 426 dev examples (dev.tsv)
- 5 427 test examples (test.tsv)

These files have no header row and have the following columns:
- text
- comma-separated list of emotion ids (the ids are indexed based on the
order of emotions in emotions.txt)
- id of the comment

The emotion categories are: admiration, amusement, anger, annoyance, 
approval, caring, confusion, curiosity, desire, disappointment, disapproval, 
disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, 
nervousness, optimism, pride, realization, relief, remorse, sadness, surprise.

Source code -> [github](https://github.com/google-research/google-research/tree/master/goemotions)

## External sentiment / emotion resources
### SentiWordNet 3.0 (EN)
SentiWordNet is a lexical resource for opinion mining which assigns to each 
synset of Princeton WordNet three sentiment scores: 
positivity, negativity, objectivity. 
See papers: [intro](https://aclanthology.org/L06-1225/), 
[version 3.0](https://aclanthology.org/L10-1531/).

### Polish Wordnet (PL)
Polish lexical relation graph. Consists of plWordNet Emo extensions with polarity dimensions for each sense.
