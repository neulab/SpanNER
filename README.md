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



## Data Preprocessing

The dataset needs to be preprocessed, before running the model.
We provide `dataprocess/bio2spannerformat.py` for reference, which gives the CoNLL-2003 as an example. 
First, you need to download datasets, and then convert them into BIO2 tagging format. We provided the CoNLL-2003 dataset with BIO format in `data/conll03_bio`, and its preprocessed format dataset `data/conll03`.

The download links of the datasets used in this work are shown as follows:
- [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
- [WNUT-2016](http://noisy-text.github.io/2016/ner-shared-task.html)
- [WNUT-2017](http://noisy-text.github.io/2017/emerging-rare-entities.html)



## Prepare Models

For English Datasets, we use [BERT-Large](https://github.com/google-research/bert)
For Dutch and Spanish Datasets, we use [BERT-Multilingual-Base](https://huggingface.co/bert-base-multilingual-uncased)






## How to run?

Here, we give CoNLL-2003 as an example. You may need to change the `DATA_DIR`, `PRETRAINED`, `dataname`, `n_class` to your own dataset path, pre-trained model path, dataset name, and the number of labels in the dataset, respectively.

`./run_conll03_spanner.sh`



## System Combination

### Base Model
We provided 12 base model result-files of CoNLL-2003 dataset in `combination/results`.
More base model's result-files can be download from [ExplainaBoard-download](http://explainaboard.nlpedia.ai/download.html).

### Combination
Put your result-files with different models in `data/results` folder, then run:

`python comb_voting.py`. 

Here, we provided four system combination methods, including: SpanNER, Majority voting (VM), Weighted voting base on overall F1-score (VOF1), Weighted voting base on class F1-score (VCF1).







