## SpanNER: Named EntityRe-/Recognition as Span Prediction

This repository contains the code for our paper [SpanNER: Named EntityRe-/Recognition as Span Prediction](https://arxiv.org/pdf/2106.00641v1.pdf).

### Overview

We investigate complementary advantages of systems based on different paradigms: span prediction model and sequence labeling framework. We then reveal that span prediction, simultaneously, can serve as a system combiner to re-recognize named entities from different systems’ outputs. We experimentally implement 154 systems on 11
datasets, covering three languages, comprehensive results show the effectiveness of span prediction models that both serve as base NER systems and system combiners.

<!-- Two roles of span prediction models (boxes in blue): 
* as a base NER system 
* as a system combiner. -->

<div  align="center">
 <img src="pic/spanner.png" width = "600" alt="d" align=center />
<div>


### Requirements

- `python3`
- `PyTorch`
- `pytorch-lightning`

Run the following script to install the dependencies,
- `pip3 install -r requirements.txt`


### How to run?

`./run_conll03_spanner.sh`


### Data Preprocessing

The code for data preprocessing are shown on the folder: `dataprocess`.

Given the `path` of your datasets with bio format, and the path to store the datasets with the new format. And run:
- `python3 bio2mrcformat.py`




