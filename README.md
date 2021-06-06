## SpanNER: Named EntityRe-/Recognition as Span Prediction

Two roles of span prediction models (boxes in blue): 
* as a base NER system 
* as a system combiner.


<!-- <img src="https://hub.fastgit.org/neulab/SpanNER/blob/main/pic/spanner.jpg" width="200" height="200" alt="ff"/><br/> -->

<img src="https://github.com/neulab/SpanNER/blob/master/pic/spanner.jpg" width="200" height="200" alt="test"/><br/>



<!-- ![](pic/spanner.jpg) -->





This repository contains the code for our paper [SpanNER: Named EntityRe-/Recognition as Span Prediction](https://arxiv.org/pdf/2106.00641v1.pdf).

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




