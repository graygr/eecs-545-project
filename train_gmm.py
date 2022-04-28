from features import *
from sklearn.mixture import GaussianMixture

import pickle

fname = 'complex-fg'

def main():

    classifier = GaussianMixture(n_components=2, max_iter=10000, covariance_type='full')

    with open('./pickle/' + fname[:-3] + '_' + fname[-2:] + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
    classifier.fit(features)

    with open('./pickle/gmm_' + fname[:-3] + '_' + fname[-2:] + '.pkl', 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == '__main__':
    main() 