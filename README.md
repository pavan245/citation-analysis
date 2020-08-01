# Citation Intent Classification 
Project repo for Computational Linguistics Team Lab at the University of Stuttgart.

## Introduction
This repository contains code and datasets for classifying citation intents in research papers.

We implemented 3 classifiers and evaluated on test dataset:

 - Perceptron Classifier - Baseline model (Implemented from scratch)
 - Feedforward Neural Network Classifier (using [PyTorch](https://pytorch.org/))
 - BiLSTM + Attention with ELMo Embeddings (using [AllenNLP](https://allennlp.org/) library)

This README documentation focuses on running the code base, training the models and predictions. For more information about our project work, model results and detailed error analysis, check [this](https://www.overleaf.com/project/5f1b0e8a6d0fb80001ceb5eb) report. Slides from the mid-term presentation are available [here](/presentation.pdf).<br/>
For more information on the Citation Intent Classification in Scientific Publications, follow this [link](https://arxiv.org/pdf/1904.01608.pdf) to the original published paper and their [GitHub repo](https://github.com/allenai/scicite)

## Environment & Setup
It's recommended to use **Python 3.5 or greater**. Now we can install and create a Virtual Environment to run this project.

#### Installing virtualenv
```shell
python3 -m pip install --user virtualenv
```
#### Creating a virtual environment
**venv** (for Python 3) allows us to manage separate package installations for different projects.
```shell
python3 -m venv citation-env
```
#### Activating the virtual environment
Before we start installing or using packages in the virtual environment we need to _activate_ it.
```shell
source citation-env/bin/activate
```
#### Leaving the virtual environment
To leave the virtual environment, simply run:
```shell
deactivate
```

After activating the Virtual Environment, the console should look like this:
```shell
(citation-env) [user@server ~]$ 
```
#### Cloning the Repository
```shell
git clone https://github.com/yelircaasi/citation-analysis.git
```
Now change the current working directory to the project root folder (`> cd citation-analysis`)
**Note:** Stay in the Project root folder while running all the experiments.

#### Installing Pacakages
Now we can install all the packages required to run this project, available in [requirements.txt](/requiements.txt) file.
```shell
(citation-env) [user@server citation-analysis]$ pip install -r requirements.txt
```
#### Environment Variable for Saved Models Path
```shell
export SAVED_MODELS_PATH=/mount/arbeitsdaten/studenten1/team-lab-nlp/mandavsi_rileyic/saved_models
```

## Data
We have 3 different intents/classes in the dataset:

 - background (background information)
 - method (use of methods)
 - result (comparing results)

**Dataset Class distribution:**
|  | background | method | result |
|:---|:---:|:---:|:---:|
| train | 4.8 K | 2.3 K | 1.1 K |
| dev | 0.5 K | 0.3 K | 0.1 K |
| test | 1 K | 0.6 K | 0.2 K |

## Methods (Classification)
### 1) Perceptron Classifier (Baseline Classifier)
We implemented [Perceptron](https://en.wikipedia.org/wiki/Perceptron) as a baseline classifier, from scratch (including evaluation). Perceptron is an algorithm for supervised learning of classification. It's a linear and binary classifier, which means it can only decide whether or not an input feature belongs to some specific class and  it's only capable of learning linearly separable patterns.
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

### Running the Model
```shell
(citation-env) [user@server citation-analysis]$ python3 -m testing.model_testing
```
  
[Link](/testing/model_testing.py) to the test source code. All the Hyperparameters can be modified to experiment with.
  
### Evaluation  
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
<img src="/plots/perceptron/confusion_matrix_plot.png?raw=true" width="500" height = "375" alt = "Confusion Matrix Plot" />

### 2) Feedforward Neural Network (using PyTorch)
A feed-forward neural network classifier with a single hidden layer containing 9 units. While a feed-forward neural network is clearly not the ideal architecture for sequential text data, it was of interest to add a sort of second baseline and examine the added gains (if any) relative to a single perceptron. The input to the feedforward network remained the same; only the final model was suitable for more complex inputs such as word embeddings.

Check this feed-forward model source [code](/classifier/linear_model.py) for more details.

### 3) BiLSTM + Attention with ELMo (AllenNLP Model)
The Bi-directional Long Short Term Memory (BiLSTM) model built using the [AllenNLP](https://allennlp.org/) library. For word representations, we used 100-dimensional [GloVe](https://nlp.stanford.edu/projects/glove/) vectors trained on a corpus of 6B tokens from Wikipedia. For contextual representations, we used [ELMo](https://allennlp.org/elmo) Embeddings which have been trained on a dataset of 5.5B tokens. This model uses the entire input text, as opposed to selected features in the text, as in the first two models. It has a single-layer BiLSTM with a hidden dimension size of 50 for each direction. 

We used AllenNLP's [Config Files](https://guide.allennlp.org/using-config-files) to build our model, just need to implement a model and a dataset reader (with a JSON Config file).

Our BiLSTM AllenNLP model contains 4 major components:

 1. Dataset Reader - [CitationDatasetReader](/utils/reader.py)
	 - It reads the data from the file, tokenizes the input text and creates AllenNLP `Instances` 
	 - Each `Instance` contains a dictionary of `tokens` and `label`
 2. Model - [BiLstmClassifier](/calssifier/nn.py)
	 - The model's `forward()` method is called for every data instance by passing `tokens` and `label`
	 - The signature of `forward()` needs to match with field names of the `Instance` created by the DatasetReader
	 - The `forward()` method finally returns an output dictionary with the predicted label, loss, softmax probabilities and so on...
 3. Config File - [basic_model.json](configs/basic_model.json?raw=true)
	 - The AllenNLP Configuration file takes the constructor parameters for various objects (Model, DatasetReader, Predictor, ...)
	 - We can provide a number of Hyperparameters in this Config file.
		 - Depth and Width of the Network
		 - Number of Epochs
		 - Optimizer & Learning Rate
		 - Batch Size
		 - Dropout
		 - Embeddings
	- All the classes that config file uses must register using decorators (Ex: `@Model.register('bilstm_classifier'`).
 4. Predictor - [IntentClassificationPredictor](/testing/intent_predictor.py)
	 - AllenNLP uses `Predictor`, a wrapper around the trained model, for making predictions.
	 - The Predictor uses a pre-trained/saved model and dataset reader to predict new Instances

### Running the Model
AllenNLP provides `train`, `evaluate` and `predict` commands to interact with the models from command line.

#### Training
```shell
$ allennlp train \
    configs/basic_model.json \
    -s $SAVED_MODELS_PATH/experiment_10 \
    --include-package classifier
```
We ran a few experiments on this model, the configurations, results and archived models are available in `SAVED_MODELS_PATH` directory

### Evaluation
To evaluate the model, simply run:
```shell
$ allennlp evaluate \
    $SAVED_MODELS_PATH/experiment_4/model.tar.gz \
    data/jsonl/test.jsonl \
    --cuda-device 3 \
    --include-package classifier
```

### Predictions
To make predictions, simply run:
```shell
$ allennlp predict \
    $SAVED_MODELS_PATH/experiment_4/model.tar.gz \
    data/jsonl/test.jsonl \
    --cuda-device 3 \
    --include-package classifier
    --predictor citation_intent_predictor
```

### Results
<img src="/plots/bilstm_model/confusion_matrix_plot.png?raw=true" width="500" height = "375" alt = "Confusion Matrix Plot" />

## References