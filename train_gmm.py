from features import *
from sklearn.mixture import GaussianMixture

import pickle

fpath = "AMOS2019-master/assets/data/simple-fg.mp4"

def main():

    features = []

    vr = cv2.VideoCapture(fpath)
    if not vr.isOpened():
        raise Exception("Error opening video stream or file")
    i = 0
    f_frames = np.zeros((num_frames, 1080, 1920))

    while(vr.isOpened()):
        c_frame = np.zeros((1080, 1920))
        ret, frame_in = vr.read()
        i += 1
        if ret:
            f_frames[i % num_frames] = cv2.cvtColor(frame_in, cv2.COLOR_RGB2GRAY)
            ret, f_frames[i % num_frames] = cv2.threshold(f_frames[i % num_frames], 1, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), 'uint8')
            f_frames[i % num_frames] = cv2.dilate(f_frames[i % num_frames], kernel, iterations=1)

            # Combine current frame buffer
            for j in range(num_frames):
                c_frame = np.add(c_frame, f_frames[j])

            # Rescale image to be in bounds
            c_frame = cv2.convertScaleAbs(c_frame)

            # Extract contours from fused image
            contours, hierarchy = cv2.findContours(c_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract features from contours
            features += extract(contours)
            if i%100 == 0:
                print('frame ', i)
        else:
            break

    vr.release()

    classifier = GaussianMixture(n_components=2, max_iter=10000, covariance_type='full')
    print('fitting')
    classifier.fit(features)

    with open('./pickle/gmm_simple_fg.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('./pickle/simple_fg_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    # print('classifying')
    # y_hat = classifier.predict(features)
    # print(y_hat.size)
    # raise


if __name__ == '__main__':
    main() 

# features = np.array(features)
    # print(features.shape)
    # preds = gmm.predict(features)
    # fig, ax = plt.subplots(figsize=(9, 6))
    # ax.scatter(features[:, 0], features[:, 1], c=preds, cmap='rainbow', s = 18)
    # ax.set_title('simple-fg.mp4, frame 80')
    # ax.set_xlabel('area')
    # ax.set_ylabel('perimeter')
    # fig.savefig('test.png')
    # cv2.imwrite("test2.png", c_frame)
    # break