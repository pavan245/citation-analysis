import numpy as np
from itertools import chain
from utils.csv import read_csv_file


def read_csv_nn_dev(scicite_dir=None):

	dev_file_path = 'data/tsv/dev.tsv'
	dev_raw = read_csv_file(dev_file_path, '\t')

	features = [x.features for x in dev_raw]
	features_unique = list(set(chain.from_iterable(features)))
	nobs = len(features)
	nfeats = len(features_unique)

	X_dev = np.zeros((nobs, nfeats))

	for j in range(nfeats):
		f = features_unique[j]
		for i in range(nobs):
			if f in features[i]:
				X_dev[i,j] = 1

	y_dev_raw = np.array([x.true_label for x in dev_raw])
	y_unique = sorted(list(set(y_dev_raw)))
	y_dim = len(y_unique)
	y_dev = np.zeros((nobs,y_dim))

	for j in range(y_dim):
		y_dev[:,j] = y_dev_raw == y_unique[j]

	return X_dev, y_dev

