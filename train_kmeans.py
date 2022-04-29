from features import *
from sklearn.cluster import KMeans

import pickle

fname = 'complex-bg'
kernel_flag = 1

def main():

    classifier = KMeans(n_clusters=2, max_iter=10000)

    with open('./pickle/' + fname[:-3] + '_' + fname[-2:] + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
        if kernel_flag != 0:
            features = np.reshape(np.linalg.norm(features, axis=1), (-1, 1))
    
    classifier.fit(features)

    if kernel_flag == 0:
        with open('./pickle/kmeans_' + fname[:-3] + '_' + fname[-2:] + '.pkl', 'wb') as f:
            pickle.dump(classifier, f)
    elif kernel_flag == 1:
        with open('./pickle/kmeans_kernel_' + fname[:-3] + '_' + fname[-2:] + '.pkl', 'wb') as f:
            pickle.dump(classifier, f)

if __name__ == '__main__':
    main()