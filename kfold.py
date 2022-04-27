# from features import *
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def kFold(k, classifier_type, fname):
    classifier = None

    if classifier_type == 'gmm':
        classifier = GaussianMixture(n_components=2, max_iter=10000, covariance_type='full')
    elif classifier_type == 'kmeans':
        classifier = KMeans(n_clusters=2, max_iter=10000)

    features = None
    num_features = None
    with open('./pickle/' + fname[:-3] + '_' + fname[-2:] + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('./pickle/' + fname[:-3] + '_' + fname[-2:] + '_num_features.pkl', 'rb') as f:
        num_features = pickle.load(f)

    features = np.array(features)
    N = len(features)
    num_debris = np.zeros(len(num_features))
    idxs_shuffled = np.arange(0, N, 1, dtype=int)
    np.random.shuffle(idxs_shuffled)
    B = N // k
    result = np.zeros(len(features))
    for i in range(k):
        remainder_idx = np.append(idxs_shuffled[:i*B], idxs_shuffled[(i+1)*B:])
        test_slice_idx = idxs_shuffled[i*B : (i+1)*B]
        classifier.fit(features[remainder_idx])
        # y_hat = classifier.predict(features[test_slice_idx])
        # result = np.zeros(N)
        # result[test_slice_idx] = y_hat
        y_hat = classifier.predict(features)
        y_hat[remainder_idx] = 0
        result += y_hat

    cur = 0
    for j, num_feature in enumerate(num_features):
        num_debris[j] += [sum(result[cur : cur + num_feature])]
        cur += num_feature
    # print(num_debris)
    id = np.arange(0, len(num_features), 1, dtype=int)
    res = np.array([np.array(id), np.array(num_features), np.array(num_debris)], dtype=int).T
    return res
# kFold(10, 'gmm', fname = "simple-bg")