import numpy as np
from itertools import chain
from utils.csv import read_csv_file

# TODO: clean up, transform into class, allow for command-line arguments

def read_csv_nn(scicite_dir=None):

	train_file_path = 'data/tsv/train.tsv'
	test_file_path = 'data/tsv/test.tsv'
	train_raw = read_csv_file(train_file_path, '\t')

	features = [x.features for x in train_raw]
	features_unique = list(set(chain.from_iterable(features)))
	nobs = len(features)
	nfeats = len(features_unique)

	X_train = np.zeros((nobs, nfeats))

	for j in range(nfeats):
		f = features_unique[j]
		for i in range(nobs):
			if f in features[i]:
				X_train[i,j] = 1

	y_train_raw = np.array([x.true_label for x in train_raw])
	y_unique = sorted(list(set(y_train_raw)))
	y_dim = len(y_unique)
	y_train = np.zeros((nobs,y_dim))

	for j in range(y_dim):
		y_train[:,j] = y_train_raw == y_unique[j]

	test_raw = read_csv_file(test_file_path, '\t')
	features = [x.features for x in test_raw]
	#features_unique = list(set(chain.from_iterable(features)))
	nobs = len(features)
	nfeats = len(features_unique)

	X_test = np.zeros((nobs, nfeats))
	for j in range(nfeats):
		f = features_unique[j]
		for i in range(nobs):
			if f in features[i]:
				X_test[i,j] = 1

	return X_train, y_train, X_test


