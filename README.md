# Quora Insincere Questions Classification
#### Data science competition at Kaggle.com
#### https://www.kaggle.com/c/quora-insincere-questions-classification
#### Top 2% solution
Solution for Quora Insincere Questions Classification, data science competition at Kaggle.com.
Goal of the competition was to built algorithm that identifies toxic questions asked at quora.com. 
Algorithms were evaluated by F1 score.

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

## Solution description
My final solution is an ensemble of 5 neural network models: 4 Recurrent Neural Network models and
1 Neural Network model with fully connected layers.

Each RNN model used different pretrained embeddings: glove.840B.300d, GoogleNews-vectors-negative300,
paragram_300_sl999, wiki-news-300d-1M.

Executing script ens_main.py with default arguments runs final algorithm from beginning to the end.


