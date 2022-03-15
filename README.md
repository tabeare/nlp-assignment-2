# NLP Assignment 2 - BERT Classifier for Aspect Based Sentiment Analysis
Natural Language Processing course at CentraleSupélec, Spring 2022  
Joshua Hiepler, Caroline Maurent, Hugo Mauron, Tabea Redl  

## Running instructions
To start the training and evaluation of a model, run ```python tester.py``` from the source (src) folder. 

### Structure
```
.
├── _data
│   ├── devdata.csv       (Validation data - used for hyperparameter tuning)
│   └── traindata.csv     (Training data)
├── _src
│   ├── classifier.py     (Main class starting the training and storing the model weights)
│   ├── bert.py           (Classes implementing model architecture)
│   ├── dataset.py        (Class implementing Dataset that can be used by pytorch Dataloader)
│   ├── tokenizer.py      (Class for tokenization of the sentences)
│   └── tester.py         (Script for running training and testing provided in the assignement)
├── Exercise2_Instructions_2022.pdf
└── README.md
```

## Problem Description
The goal of this project is to predict the polarity (negative, neutral or positive) for a given aspect category that is associated with a given term in a sentence. For example, for the sentence "Wait staff is blantently unappreciative of your business but its the best pie on the UWS!" the goal is to predict for the term 'Wait staff' with aspect category 'SERVICE#GENERAL', that the polarity is negative.

## Description of the model
Our model is building on a pretrained BERT transformers model, first presenented by (Devlin et al. 2018)[https://arxiv.org/abs/1810.04805]. We use the BERT model as an encoder of the sentences and use the encoded sentences with a further classification model to predict the polarities. 
For the ABS classification, we build a local context focus mechanism model, which was presented by (Zeng et al. 2019)[https://mdpi-res.com/d_attachment/applsci/applsci-09-03389/article_deploy/applsci-09-03389-v3.pdf]. This model internally uses a BERT Self Attention with multiple attention heads.

## Results
Running the ```tester.py``` file, our model achieved the following results:

Dev accs: [86.97, 87.23, 86.44, 87.23, 86.97]

Test accs: [-1, -1, -1, -1, -1]

Mean Dev Acc.: 86.97 (0.29)

Mean Test Acc.: -1.00 (0.00)

Exec time: 1590.73 s. ( 318 per run )
