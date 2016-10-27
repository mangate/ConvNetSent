# Convolutional Neural Network for Sentence Classification, in TensorFlow
This is a full (slightly modified and extended) TensorFlow implementation of the model presented by Kim in [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181). 

With this code you can reproduce almost all results presented by Kim and the results we present in our [Word Embeddings and Their Use In Sentence Classification Tasks](https://arxiv.org/abs/1610.08229) paper.

## Features:
- Supports Random, Static, and non-Static modes.
- Runs on (almost) all datasets from the original article + 2 from [here](https://arxiv.org/abs/1510.03820) + 2 new ones. 
- Optimizer changed to ADAM, improving training time and some of the results.
- Support for L2 loss added.

## Credits
If you're using this code please make sure you cite both following papers:
- Kim's paper:
```
@article{Kim2014ConvNetSent,
  Author = {Kim, Yoon},
  Journal = {arXiv preprint arXiv:1408.5882},
  Title = {Convolutional Neural Networks for Sentence Classification},
  Year = {2014}
}
```
- Our paper:
```
@article{Man2016WordEmbbed,
  Author = {Mandelbaum, Amit and Shalev, Adi},
  Journal = {arXiv preprint arXiv:1610.08229},
  Title = {Word Embeddings and Their Use In Sentence Classification Tasks},
  Year = {2016}
}
```
Also, small parts of the code were taken from:
- [Yoon Kim's git repository](https://github.com/yoonkim/CNN_sentence)
- [Denny Britz's git repository](https://github.com/dennybritz/cnn-text-classification-tf)
We'd like to thank them for that.

## Requirements
- Python (2.7)
- NumPy
- TensorFlow (>=0.8)
- Pandas

## Usage
### Preperation:
1) Clone the repository recursively to get all folder and subfolders  
2) Download Google's word embeddings binary file from [https://code.google.com/p/word2vec/](https://code.google.com/archive/p/word2vec/) extract it, and place it under `data/` folder  
### Running:
1) Choose the dataset you want to run on by uncommenting it in the last section of `sentence_convnet_final.py`  
2) Run `python sentence_convnet_final.py --static <True/False> --random <True/False>`  
Below is a list of modes and appropriate flags:
- Random: `--static False --Random True`
- Static: `--static True --Random False`
- Non-Static: `--static False --Random False`
