from features import *
from sklearn.cluster import KMeans

import pickle

fname = 'simple-fg'

def main():

    classifier = KMeans(n_clusters=2, max_iter=10000)

    with open('./pickle/' + fname[:-3] + '_' + fname[-2:] + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
    classifier.fit(features)

    with open('./pickle/kmeans_' + fname[:-3] + '_' + fname[-2:] + '.pkl', 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == '__main__':
    main() 