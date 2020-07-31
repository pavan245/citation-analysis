# Citation Intent Classification 
Project repo for Computational Linguistics Team Lab at the University of Stuttgart.

## Introduction
This repository contains code and datasets for classifying citation intents in research papers.

We implemented 3 different classifiers and evaluated the results:

 - Perceptron Classifier - Baseline (Implemented from scratch)
 - Feedforward Neural Network Classifier (using [PyTorch](https://pytorch.org/))
 - BiLSTM + Attention with ELMo Embeddings (using [AllenNLP](https://allennlp.org/) library)

This README documentation focuses on running the code base, training the models and predictions. For more information about our project work and detailed error analysis, check [this](https://www.overleaf.com/project/5f1b0e8a6d0fb80001ceb5eb) report. <br/>
For more information on the Citation Intent Classification in Scientific Publications, follow this [link](https://arxiv.org/pdf/1904.01608.pdf) to the original published paper and the [GitHub repo](https://github.com/allenai/scicite)

## Environment & Setup
TODO

## Data
We have 3 different intents/classes in the dataset:

 - background (background information)
 - method (use of methods)
 - result (comparing results)

**Dataset Class distribution:**
|  | background | method | result |
|:---|:---:|:---:|:---:|
| train | 4.8 K | 2.3 K | 1.1 K |
| test | 1 K | 0.6 K | 0.2 K |

## Methods (Classification)
### 1) Perceptron Classifier (Baseline Classifier)
We implemented [Perceptron](https://en.wikipedia.org/wiki/Perceptron) as a baseline classifier, from scratch (including evaluation). Perceptron is an algorithm for supervised learning of classification. It's a Linear and a Binary Classifier, which means it can only decide whether or not an input feature belongs to some specific class and also it's only capable of learning linearly separable patterns.
```python
class Perceptron:
  def __init__(self, label: str, weights: dict, theta_bias: float):
  def score(self, features: list):
  def update_weights(self, features: list, learning_rate: float, penalize: bool, reward: bool):

class MultiClassPerceptron:
  def __init__(self, epochs: int = 5000, learning_rate: float = 1, random_state: int = 42)
  def fit(self, X_train: list, labels: list)
  def predict(self, X_test: list)
```
Since we have 3 different classes for Classification, we create a Perceptron object for each class. Each Perceptron has score and update functions. During training, for a set of input features it takes the score from the Perceptron for each label and assigns the label with max score(for all the data instances). It compares the assigned label with the true label and decides whether or not to update the weights (with some learning rate).

Check the source [code](/classifier/linear_model.py) for more details on the implementation of Perceptron Classifier.

#### Running the Model

> `(citation-env) [user@server citation-analysis]$ python -m testing.model_testing`
  
[link](/testing/model_testing.py) to the source code. All the Hyperparameters can be modified to experiment with.
  
**Evaluation**  
we used ***f1_score*** metric for evaluation of our baseline classifier.
  
> F1 score is a weighted average of Precision and Recall(or Harmonic Mean between Precision and Recall). 
> The formula for F1 Score is:  
> F1 = 2 * (precision * recall) / (precision + recall)  
  
```python  
eval.metrics.f1_score(y_true, y_pred, labels, average)  
```  
**Parameters**:
**y_true** : 1-d array or list of gold class values    
**y_pred** : 1-d array or list of estimated values returned by a classifier    
**labels** : list of labels/classes    
**average**: string - [None, 'micro', 'macro'] If None, the scores for each class are returned.

[Link](/eval/metrics.py) to the metrics source code.

### Results
<img src="/plots/perceptron/confusion_matrix_plot.png?raw=true" width="400" height = "300" alt = "Confusion Matrix Plot" />