# ai-essay-scorer

Simple NLP project with student essay texts scoring.

## Project Statement

### Problem Statement

The task is to develop a service that automatically scores student essay texts
on a discrete scale from 1 to 6. This service can help reduce the workload on
educators by providing preliminary essay scores (the teacher would only need to
validate the score). This is particularly relevant for huge open online courses,
where a single teacher may be responsible for many students. Additionally the
service can be used as a second opinion during the grading process to mitigate
the influence of subjective factors and personal bias.

### Input and Output Data Format

Input – raw essay text, output – essay score on discrete scale from 1 to 6.

### Metrics

Key metrics: F1-score and PR-AUC. These metrics provide a comprehensive and
robust evaluation for a classification task. They are particularly well-suited
for this project as they are stable in the presence of class imbalance, which is
a potential issue in the training dataset. Goals: F1-score >= 0.65 and PR-AUC >=
0.7.

### Validation

The original dataset will first be split into a training set and hold-out test
set with a ratio 80%/20%. The resulting training dataset will be split again
into a training subset and validation subset with ratio 85%/15%. The dataset
splitting will be performed using the train_test_split function from
scikit-learn library, with stratification on the target score column. To ensure
splitting reproducibility, a fixed random seed 42 will be used. Model validation
will be performed after each training epoch on the validation set. The final
trained model will be evaluated on the held-out test set to report the final
metrics.

### Data

The project will use training dataset from the kaggle competition (only the
training set id used as it is labeled):
https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data
The dataset contains 17307 samples with 3 columns: assey_id, full_text and
score. The target variable is score contains 6 unique classes (integer scores
from 1 to 6). Potential challenges: Class imbalance Topic dependency: model
might perform poorly on essays with highly exotic topics that unpresented in
dataset

Limitations:

1. The dataset consists of essay only on english. The trained model will not be
   applicable for scoring essays in other languages.

### Modeling

#### Baseline

The baseline solution will be a logistic regression classifier trained on
features extracted from the essay text using tf-idf.

#### Main model

The primary model will be a pre-trained deberta-v3-large. We will perform
fine-tuning on our essay dataset to adapt it for a specific task of multi-class
score classification.

#### Deployment

The model will be deployed as a REST API microservice with Triton inference
server and Docker.

## Setup

### 0. Install project

Requires python 3.13 or higher.

```bash
# clone and enter the repo
git clone git@github.com:yegerless/ai-essay-scorer.git
cd ai-essay-scorer

# init venv and install project dependencies
poetry install

# init precommit hook with pre-commit
poetry run pre-commit install

# before commit changes run pre-commit -- all hooks should be green
poetry run pre-commit run -a
```

### 1. Download project dataset

The DVC is used for data versioning, but project doesn't have any data remote
storage, only kaggle dataset page. You can download dataset from kaggle with
command below, after that you can init local imitation of remote DVC storage.

```bash
# download dataset from kaggle
poetry run python ./ai_essay_scorer/commands.py download_data --kaggle_dataset yegerless/student-essay-with-scores-dataset --dataset_file raw_data.csv --path_to_save ./data/raw_data.csv

# init local imitation of remote DVC storage
poetry run dvc push
```
