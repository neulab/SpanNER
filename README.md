## SpanNER: Named EntityRe-/Recognition as Span Prediction

This repository contains the code for our paper [SpanNER: Named EntityRe-/Recognition as Span Prediction](https://arxiv.org/pdf/2106.00641v1.pdf).

The model designed in this work has been deployed into [ExplainaBoard](http://explainaboard.nlpedia.ai/leaderboard/task-ner/index.php).

## Overview

We investigate complementary advantages of systems based on different paradigms: span prediction model and sequence labeling framework. We then reveal that span prediction, simultaneously, can serve as a system combiner to re-recognize named entities from different systems’ outputs. We experimentally implement 154 systems on 11
datasets, covering three languages, comprehensive results show the effectiveness of span prediction models that both serve as base NER systems and system combiners.

<!-- Two roles of span prediction models (boxes in blue): 
* as a base NER system 
* as a system combiner. -->

<div  align="center">
 <img src="pic/spanner.png" width = "550" alt="d" align=center />
</div>


## Requirements

- `python3`
- `PyTorch`
- `pytorch-lightning`

Run the following script to install the dependencies,
- `pip3 install -r requirements.txt`


## How to run?

`./run_conll03_spanner.sh`


## Data Preprocessing

The dataset needs to be preprocessed, before running the model.
We provide `dataprocess/bio2spannerformat.py` for reference, which gives the CoNLL-2003 as an example.
First, you need to download datasets, and then convert them into BIO2 tagging format. The download link of the datasets used in this work is shown as follows:
- [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
- [WNUT-2016](http://noisy-text.github.io/2016/ner-shared-task.html)
- [WNUT-2017](http://noisy-text.github.io/2017/emerging-rare-entities.html)



## Prepare Models

For English Datasets, we use [BERT-Large](https://github.com/google-research/bert)

For Dutch and Spanish Datasets, we use [BERT-Multilingual-Base](https://huggingface.co/bert-base-multilingual-uncased)








