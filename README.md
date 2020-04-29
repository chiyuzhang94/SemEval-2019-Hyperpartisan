# SemEval 2019 Task 4 Hyperpartisan
* Competition Details: https://pan.webis.de/semeval19/semeval19-web/

## Data
* Sample data is in [./data/](https://github.com/chiyuzhang94/hyperpartisan/tree/master/data)





## Steps:
* Use Keras to build a RNN or LSTM, understand how to process input data. [Link](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
* Text Classification Using CNN, LSTM and Pre-trained Glove Word Embeddings [Part1](https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b)  [Part3](https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa)
* Hyperparemeters setting: sequence length, size of word vocabulary, learning rate, number of batch etc.
* Pytoch implement RNN and LSTM. [Morvan pytorch](https://morvanzhou.github.io/tutorials/machine-learning/torch/), [Pytorch RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#), [Pytorch LSTM](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py)
* Pytoch implement CNN [CNN](https://github.com/Shawn1993/cnn-text-classification-pytorch)
* Model optimization
* Visulization

## Related References:
* [Beyond Binary Labels: Political Ideology Prediction of Twitter Users](http://www.aclweb.org/anthology/P17-1068)
* [A Stylometric Inquiry into Hyperpartisan and Fake News](http://aclweb.org/anthology/P18-1022)
* [Classification of Moral Foundations in Microblog Political Discourse](http://aclweb.org/anthology/P18-1067)
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
* [Fake News Detection on Social Media: A Data Mining Perspective](http://delivery.acm.org.ezproxy.library.ubc.ca/10.1145/3140000/3137600/p22-shu.pdf?ip=142.103.160.110&id=3137600&acc=ACTIVE%20SERVICE&key=FD0067F557510FFB%2E26E2C50968A06846%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1539896140_86ced20cfa1d864d0da1016a9d3fbc50)
* [Satirical News Detection and Analysis using Attention Mechanism and Linguistic Features](https://arxiv.org/pdf/1709.01189.pdf)
* [Recurrent Convolutional Neural Networks for Text Classification](https://zhuanlan.zhihu.com/p/21253220) 
* [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)
* [Attention is All you Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
* [A Convolutional Attention Model for Text Classification](https://link.springer.com/chapter/10.1007/978-3-319-73618-1_16)
* [Attention-based LSTM for Aspect-level Sentiment Classification](https://aclweb.org/anthology/D16-1058)
* [Predicting Affective Content in Tweets with Deep Attentive RNNs and Transfer Learning](http://www.aclweb.org/anthology/S17-2126)
* [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

Citation:
```
@inproceedings{zhang2019ubc,
  title={UBC-NLP at SemEval-2019 Task 4: Hyperpartisan News Detection With Attention-Based Bi-LSTMs},
  author={Zhang, Chiyu and Rajendran, Arun and Abdul-Mageed, Muhammad},
  booktitle={Proceedings of the 13th International Workshop on Semantic Evaluation},
  pages={1072--1077},
  year={2019}
}
```
