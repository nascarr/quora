# Quora Insincere Questions Classification
#### Data science competition at Kaggle.com
#### https://www.kaggle.com/c/quora-insincere-questions-classification
#### Top 2% solution, Silver Medal
## Introduction
This repository contains my solution for Quora Insincere Questions Classification, data science competition at Kaggle.com.
Goal of the competition was to built algorithm that identifies toxic/insincere questions asked at quora.com. 
Algorithms were evaluated by F1 score for insincere questions.

My solution scored F1 = 0.704.

## Competition description
In this competition you will be predicting whether a question asked on Quora is sincere or not.

An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:

* Has a non-neutral tone
    * Has an exaggerated tone to underscore a point about a group of people
    * Is rhetorical and meant to imply a statement about a group of people
* Is disparaging or inflammatory
    * Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
    * Makes disparaging attacks/insults against a specific person or group of people
    * Based on an outlandish premise about a group of people
    * Disparages against a characteristic that is not fixable and not measurable
* Isn't grounded in reality
    * Based on false information, or contains absurd assumptions

## Data
Models were trained on 1.3 million of questions and binary labels (Sincere/Insincere).

Some examples:
| Sincere | Insincere |
| --- | --- |
| What can you say about feminism? | Has the United States become the largest dictatorship in the world? |
| How were the Calgary Flames founded? | Which babies are more sweeter to their parents? Dark skin babies or light skin babies? |
| Should I leave my friends and find new ones? | Was AIDS a punishment from God for gay people? |
| How can I top CBSE in 6 months? | Why did the Pope say that "Muslims are a curse to this world"? |
| What does great wit mean? | Should I really feel that it's okay not to love or respect women? |

Data is imbalanced (94% sincere questions, 6% insincere questions). Labeling is noisy, since definition of 'insincere question' is subjective.

![alt text](https://github.com/nascarr/quora/blob/master/class_ratio.png?raw=true)

## Solution description
My final solution is an ensemble of 5 neural network models: 4 Recurrent Neural Network models and
1 Neural Network model with fully connected layers.

Each RNN model used different pretrained embeddings: glove.840B.300d, GoogleNews-vectors-negative300,
paragram_300_sl999, wiki-news-300d-1M.

Other important parts of the solution:
- code for automatic logging of experiments into .csv file
- class for model errors analysis (most_correct, most_incorrect, most_doubt, random_doubt, random_correct, random_incorrect)
- code for for ensembling (mean, weighted ensemble)
- 5 fold stratified cross validation
- reverse engineering of tokenizer for GoogleNews embeddings
- variable sequence length in batches
- trainable initialization of hidden state
- early stopping
- freeze embeddings first two epochs, finetune embeddings during later epochs
- learning rate decay
- experiments with model architecture (GRU/LSTM, average/max pooling, 0-2 FC layers on top of RNN output)
- hyperparameter optimization (> 1000 experiments)

## Prerequisite
See requirements.txt.
GPU is required for training.

## Train model with different hyperparameters
#### Make script executable.

``` chmod +x main.py```

#### Train RNN model with default hyperparameters.

```./main.py```

#### Test model with default hyperparameters. 

In test mode model is trained and evaluated on small datasets (1000 entries) using pretrained embedding for most common 10000 words.

```./main.py --mode test```

#### Train model with specific hyperparameters. Example.

```./main.py -hd 150 -em paragram -us 0.1 --seed 4 -s -e 10 -es 3```

#### Train model with various hyperparameters. 

See ```./main.py --help```

Output:

```
 -h, --help            show this help message and exit
  --machine {dt,kaggle}
                        Local machine: dt. Kaggle kernel: kaggle.
                        
  --mode {test,run}     Main mode: run. Test mode: test.
  --kfold KFOLD, -k KFOLD
                        K-fold cross-validation.
  --split_ratio SPLIT_RATIO [SPLIT_RATIO ...], -sr SPLIT_RATIO [SPLIT_RATIO ...]
                        Split ratio.
  --test                If present split data in train-val-test else split in
                        train-val.
  --seed SEED           Seed for data split.
  --tokenizer TOKENIZER, -t TOKENIZER
                        Tokenizer. See tokenizers.py.
  --embedding {glove,gnews,paragram,wnews} [{glove,gnews,paragram,wnews} ...], -em {glove,gnews,paragram,wnews} [{glove,gnews,paragram,wnews} ...]
                        Embedding.
  --max_vectors MAX_VECTORS, -mv MAX_VECTORS
                        Load no more than max_vectors number of embedding
                        vectors.
  --no_cache            Don't cache embeddings.
  --var_length, -vl     Variable sequence length in batches.
  --unk_std UNK_STD, -us UNK_STD
                        Standart deviation for initialization of tokens
                        without embedding vector.
  --stratified, -s      Stratified split.
  --optim {Adam,AdamW}, -o {Adam,AdamW}
                        Optimizer. See choose.py
  --epoch EPOCH, -e EPOCH
                        Number of epochs.
  --lr LR, -lr LR       Initial learning rate.
  --lrstep LRSTEP [LRSTEP ...]
                        Steps when lr multiplied by 0.1.
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size.
  --n_eval N_EVAL, -ne N_EVAL
                        Number of validation set evaluations during 1 epoch.
  --warmup_epoch WARMUP_EPOCH, -we WARMUP_EPOCH
                        Number of epochs without embedding tuning.
  --early_stop EARLY_STOP, -es EARLY_STOP
                        Stop training if no improvement during this number of
                        epochs.
  --f1_tresh F1_TRESH, -ft F1_TRESH
                        Threshold for calculation of F1-score.
  --clip CLIP           Gradient clipping.
  --model MODEL, -m MODEL
                        Model name. See models.py.
  --n_layers N_LAYERS, -n N_LAYERS
                        Number of RNN layers in model.
  --hidden_dim HIDDEN_DIM, -hd HIDDEN_DIM
                        Hidden dimension for RNN.
  --dropout DROPOUT, -d DROPOUT
                        Dropout probability.
```

## Run final solution: ensemble of 5 models
```./ens_main.py```


