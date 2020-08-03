# Citation Intent Classification 
Project repo for Computational Linguistics Team Lab at the University of Stuttgart.

## Introduction
This repository contains code and datasets for classifying citation intents in research papers.

We implemented 3 classifiers and evaluated on test dataset:

 - Perceptron Classifier - Baseline model (Implemented from scratch)
 - Feedforward Neural Network Classifier (using [PyTorch](https://pytorch.org/))
 - BiLSTM + Attention with ELMo Embeddings (using [AllenNLP](https://allennlp.org/) library)

This README documentation focuses on running the code base, training the models and predictions. For more information about our project work, model results and detailed error analysis, check [this](/14-final-report-Mandava-Riley.pdf) report. Slides from our mid-term presentation are available [here](/presentation.pdf).<br/>
For more information on the Citation Intent Classification in Scientific Publications, follow this [link](https://arxiv.org/pdf/1904.01608.pdf) to the original published paper and their [GitHub repo](https://github.com/allenai/scicite)

## Environment & Setup
This project needs **Python 3.5 or greater**. We need to install and create a Virtual Environment to run this project.

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
Now change the current working directory to the project root folder (`> cd citation-analysis`). <br />
**Note:** Stay in the Project root folder while running all the experiments.

#### Installing Pacakages
Now we can install all the packages required to run this project, available in [requirements.txt](/requirements.txt) file.
```shell
(citation-env) [user@server citation-analysis]$ pip install -r requirements.txt
```
#### Environment Variable for Saved Models Path
Run the below line in the console, we'll use this variable later on.
```shell
export SAVED_MODELS_PATH=/mount/arbeitsdaten/studenten1/team-lab-nlp/mandavsi_rileyic/saved_models
```

## Data
This project uses a large dataset of citation intents provided by this `SciCite` [GitHub repo](https://github.com/allenai/scicite). Can be downloaded from this [link](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz). <br />
We have 3 different intents/classes in this dataset:

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
<img src="/plots/perceptron/confusion_matrix_plot.png?raw=true" width="600" height = "450" alt = "Confusion Matrix Plot" />

### 2) Feed-forward Neural Network Classifier (Baseline Classifier)
A feed-forward neural network classifier with a single hidden layer containing 9 units. While  clearly not the ideal architecture for sequential text data, the feed-forward neural network provides a second baseline. The input to the feedforward network remained the same as the perceptron; only the third model is suitable for more complex inputs such as word embeddings.
```python
class FeedForward(torch.nn.Module):
  def __init__(self, input_size: int, hidden_size: int, output_size: int):
  def forward(self, x: torch.nn.FloatTensor):
  def read_data(self):
  def fit(self, epochs: int = 100, batch_size: int = 16, lr: int = 0.01, 
          samples: tuple = (1000, 1000, 1000)):
  def predict(self):
  def shuffle(self):

```

Check the source [code](/classifier/nn_ff.py) for more details on the implementation of the feed-forward neural network.

### Running the Model
```shell
(citation-env) [user@server citation-analysis]$ python3 -m testing.ff_model_testing
```
  
[Link](/testing/ff_model_testing.py) to the test source code. All the Hyperparameters can be modified to experiment with.
  
### Evaluation  
As in the perceptron classifier, we used ***f1_score*** metric for evaluation of our baseline classifier.

### Results
<img src="/plots/ffnn_model/confusion_matrix_plot_ff.png?raw=true" width="600" height = "450" alt = "Confusion Matrix Plot" />

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
	 - This Model uses [ELMo](https://allennlp.org/elmo) deep contextualised embeddings.
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
	- All the classes that the Config file uses must register using Python decorators (for example, `@Model.register('bilstm_classifier'`).
 4. Predictor - [IntentClassificationPredictor](/classifier/intent_predictor.py)
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
We ran a few experiments on this model, the run configurations, results and archived models are available in the `SAVED_MODELS_PATH` directory. <br />
**Note:** If the GPU cores are not available, set the `"cuda_device":` to `-1` in the [config file](/configs/basic_model.json?raw=true), otherwise the available GPU Core.

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
    --include-package classifier \
    --predictor citation_intent_predictor
```

We also have an another way to make predictions without using `allennlp predict` command. This returns prediction list, softmax probabilities and more details useful for error analysis. Simply run the following command: 
```shell
(citation-env) [user@server citation-analysis]$ python3 -m testing.bilstm_predict
```
Modify [this](/testing/bilstm_predict.py) source to run predictions on different experiments. It also saves the Confusion Matrix Plot (as shown below) after prediction.

### Results
<img src="/plots/bilstm_model/confusion_matrix_plot.png?raw=true" width="600" height = "450" alt = "Confusion Matrix Plot" />

## References
[\[1\]](https://github.com/allenai/scicite) SciCite GitHub Repository<br />
This repository contains datasets and code for classifying citation intents, our poroject is based on this repository. <br /><br />
[\[2\]](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz) SciCite Dataset <br />
Large Datset of Citation Intents <br /> <br />
[\[3\]](https://allennlp.org/tutorials) AllenNLP Library.<br />
An open-source NLP research library, built on PyTorch. <br /><br />
[\[4\]](https://allennlp.org/elmo) ELMo Embeddings<br />
Deep Contextualized word representations. <br /><br />
[\[5\]](https://guide.allennlp.org/) AllenNLP Guide<br />
A Guide to Natural Language Processing With AllenNLP. <br /><br />

