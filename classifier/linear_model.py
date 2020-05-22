from utils.csv import DataInstance
from feature_extraction.features import FEATURE_LIST, THETA_BIAS_FEATURE
from collections import OrderedDict
import random


class Perceptron:

    """
    Perceptron is an algorithm for supervised learning of binary classifiers,
    which can decide whether or not an input(features) belongs to some specific class.
    It's a linear classifier, which makes predictions by combining weights with feature vector.
    """

    def __init__(self, label: str, weights: dict, theta_bias: float):
        """
        :type label: str
        :type weights: dict
        :type theta_bias: float

        :param label:  Label for the Perceptron Classifier (useful while dealing with Multi-Class Perceptron)
        :param weights: dictionary of feature name and feature weights(random number)
        :param theta_bias: value of the theta bias variable, threshold weight in other words
        """
        self.label = label
        self.weights = weights
        self.theta_bias = theta_bias

    def score(self, features: list):
        """
        This function takes the list of features as parameter and
        computes score by adding all the weights that corresponds to these features

        :type features: list

        :param features: list of features from a DataInstance
        :return: returns the computed score
        """
        score_val = 0
        for feature in features:
            score_val += self.weights[feature]

        return score_val

    def update_weights(self, features: list, learning_rate: float = 1.0, penalize: bool = False, reward: bool = False):
        """
        This function is used to update weights during the training of the Perceptron Classifier.
        It takes a list of features as parameter and updates(either increase or decrease) the
        weights for these individual features based on learning rate parameter

        :param features: list of features from Input DataInstance
        :param learning_rate: Default is 1.0
        :param penalize: If True, decreases the weights for each feature. Default is False
        :param reward: If True, increases the weights for each feature. Default is False

        - If both penalize and reward params are False, weights will not get updated.
        - If both penalize and reward are True without a learning rate(or learning rate 1),
            weights for the features remain the same.
        """
        for feature in features:
            feature_weight = self.weights[feature]
            if penalize:
                self.weights[feature] = round(feature_weight - (learning_rate * 1), 5)
            if reward:
                self.weights[feature] = round(feature_weight + (learning_rate * 1), 5)


class MultiClassPerceptron:
    """
    Perceptron is a binary classifier, can only separate between two classes.
    Multi-Class Perceptron can be used, where multiple labels can be assigned to each data instance.

    Multi-Class Perceptron creates one Perceptron Classifier for each label, while training
     it takes the score for each label(from Perceptron Classifier) and
     the label with the highest score is the predicted label

     If the predicted label is different from true label of data instance,
     this model updates the weights as follows:
        - decrease the weights for the Perceptron Classifier of predicted label (penalize)
        - increase the weights for the Perceptron Classifier of true label (reward)

     This model also shuffles the training data after each epoch.
    """
    def __init__(self, epochs: int = 5000, learning_rate: float = 1.0, random_state: int = 42):
        """
        :type epochs: int
        :type learning_rate: float
        :type random_state: int

        :param epochs: number of training iterations
        :param learning_rate: learning rate for updating weights, Default is 1
        :param random_state: random state for shuffling the data, useful for reproducing the results.
                    Default is 42.
        """
        self.random_state = random_state
        self.perceptron_dict = OrderedDict()  # contains Key : label and value : Perceptron Object for label
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X_train: list, labels: list):
        """
        This function takes the training data and labels as parameters and trains the model

        :type X_train: list[DataInstance]
        :type labels: list[str]

        :param X_train: list of training Data Instances
        :param labels: list of classes
        """

        # Check if labels parameter is empty and raise Exception
        if labels is None or len(labels) <= 0:
            raise Exception('The labels parameter must contain at least one label')

        # Check if Training Data is empty and raise Exception
        if X_train is None or len(X_train) <= 0:
            raise Exception('Training data can\'t be Empty')

        # Check the data type of training Instances
        if not isinstance(X_train, list) and not isinstance(X_train[0], DataInstance):
            raise Exception('Training Data must be a list of type DataInstance(model)')

        train_len = len(X_train)

        # Dictionary for storing label->Perceptron() objects, Create a new Perceptron object for each label
        for label in labels:
            sample_weights = get_sample_weights_with_features(theta_bias=-0.25, random_state=self.random_state)
            self.perceptron_dict[label] = Perceptron(label, sample_weights, theta_bias=-0.25)

        next_print = int(self.epochs/10)

        random.seed(self.random_state)
        random_list = [random.randint(0, train_len-1) for i in range(self.epochs)]

        # Training Iterations
        for epoch in range(self.epochs):

            if epoch >= next_print:
                print('Training Multi-Class Perceptron Classifier..... (', epoch, '/', self.epochs, ')')
                next_print = next_print + int(self.epochs/10)

            # Pick a number from random list
            inst = X_train[random_list[epoch]]

            perceptron_scores = []  # list for storing perceptron scores for each label
            for label, perceptron in self.perceptron_dict.items():
                perceptron_scores.append(perceptron.score(inst.features))

            # find the max score from the list of scores
            max_score = max(perceptron_scores)

            # find the label that corresponds to max score
            label_max_score = labels[perceptron_scores.index(max_score)]

            # if the label with max score is different from the label of this data instance,
            # then decrease the weights(penalize) for the Perceptron of label with max score
            # and increase the weights(reward) for the Perceptron of data instance label
            if inst.true_label != label_max_score:
                # decrease weights
                self.perceptron_dict[label_max_score].update_weights(inst.features, self.learning_rate, penalize=True)
                # increase weights
                self.perceptron_dict[inst.true_label].update_weights(inst.features, self.learning_rate, reward=True)

            # It's important to shuffle the list during every epoch
            random.Random(self.random_state).shuffle(X_train)

    def predict(self, X_test: list):
        """
        This function takes testing instances as parameters and assigns a predicted label.

        Takes the score from each Perceptron Classifier, label with the highest score is the predicted label

        :param X_test: list of test data instances
        :return: list of predicted labels
        """

        if X_test is None or len(X_test) <= 0:
            raise Exception('Testing Data cannot be empty')

        print('Predicting..... ')

        y_test = []
        labels = list(self.perceptron_dict.keys())
        for test_inst in X_test:
            perceptron_scores = []  # list for storing perceptron scores for each label
            for label in labels:
                perceptron_scores.append(self.perceptron_dict[label].score(test_inst.features))
            # find the max score from the list of scores
            max_score = max(perceptron_scores)

            label_max_score = labels[perceptron_scores.index(max_score)]
            y_test.append(label_max_score)

        return y_test


def get_sample_weights_with_features(theta_bias: float = 0.0, random_state: int = 42):
    """
    This function creates a dictionary with feature as a key and a random floating number (feature weight) as value.
    Weights for each feature is a floating number between -1 and 1

    :type theta_bias: float
    :type random_state: int

    :param theta_bias: value of theta bias variable
    :param random_state: random seed number for reproducing the results

    :return: returns a dictionary of random weights for each feature
    """
    weights = {THETA_BIAS_FEATURE: theta_bias}
    for feature in FEATURE_LIST:
        random.seed(random_state)
        weights[feature] = round(random.uniform(-1.0, 1.0), 5)

    return weights
