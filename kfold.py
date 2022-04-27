from features import *

def kFold(k):
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
    frame_count = []
    idxs_shuffled = np.arange(0, N, 1, dtype=int)
    np.random.shuffle(idxs_shuffled)
    B = N // k
    for i in range(k):
        remainder_idx = np.append(idxs_shuffled[:i*B], idxs_shuffled[i*B:(i+1)*B])
        test_slice_idx = idxs_shuffled[i*B : (i+1)*B]
        classifier.fit(features[remainder_idx])
        y_hat = classifier.predict(features[test_slice_idx])
        result = np.zeros(N)
        result[test_slice_idx] = y_hat

        cur = 0
        for num_feature in num_features:
            frame_count += [sum(result[cur : cur + num_feature])]
            cur += num_feature
        
    print('#####################')
    print(frame_count)
    print('#####################')