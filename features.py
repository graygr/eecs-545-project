# Read in .mp4 file, run openCV to binarize, dilate stars, fuse frames together
# then run blob detection from openCV to extract location, area, and shape. 

import cv2
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

num_frames = 5
frame_stride = 3
fname = "simple-bg"
classifier_type = 'naive'
mean_alpha = 0.9

fpath = "AMOS2019-master/assets/data/" + fname + ".mp4"
fout_name = fname + '-' + classifier_type + ".avi"

# GT file paths
gt_fpath = "groundtruth-files/" + fname + "-gt.csv"
# sfg_p = "groundtruth-files/simple-fg-gt.csv"
# sbg_p = "groundtruth-files/simple-bg-gt.csv"
# cfg_p = "groundtruth-files/complex-fg-gt.csv"
# cbg_p = "groundtruth-files/complex-bg-gt.csv"

# Helper function to read in groundtruth data files
def read_gt(fname):
    return np.genfromtxt(fname, delimiter=',', skip_header=1)

# Helper function to plot classification performance
def plot_stats(class_stats, gt_stats):
    # GT is staggered 30-60 frames, so need to interpolate in code for plotting
    gt_interpolated = []
    j = 0 # j tracks gt_stats index
    i = 1
    for frame in class_stats:
        frame_index = int(frame[0])
        # If current frame is greater than the window, then bump up the next
        if frame_index > int(gt_stats[j][0]):
            # print("i: " + str(i))
            # print("j: " + str(j))
            # print("gt_frame: " + str(int(gt_stats[j][0])))
            j += 1
        gt_interpolated.append(gt_stats[j].copy())
        gt_interpolated[i - 1][0] = frame_index

        i += 1

    # Plot
    gt_interpolated = np.array(gt_interpolated)
    plt.plot(class_stats[:, 0], class_stats[:, 2], label="# Debris Objects")
    plt.plot(gt_interpolated[:, 0], gt_interpolated[:, 2], label="# GT Debris Objects")
    plt.legend()
    plt.show()


# Helper function to compute the average area of all detected contours in the image
def avgContoursArea(contours):
    mean_area = 0
    # Find average area
    for c in contours:
        mean_area += cv2.contourArea(c)
    mean_area = mean_area / len(contours)
    return mean_area

# Draw classification boxes around outliers
def drawBoundingBoxes(contours, c_frame, features, w_vid, video_writer, past_m_features):
    # Compute mean features
    # mean_area = avgContoursArea(contours)

    classifier = None

    if classifier_type == 'gmm':
        with open('./pickle/gmm_' + fname[:-3] + '_' + fname[-2:] + '.pkl', 'rb') as f:
            classifier = pickle.load(f)
    
    elif classifier_type == 'kmeans':
        with open('./pickle/kmeans_' + fname[:-3] + '_' + fname[-2:] + '.pkl', 'rb') as f:
            classifier = pickle.load(f)

    m_features = np.zeros((4,2))

    # Compute means
    # for f in features:
    #     m_features[0,0] += f[0]
    #     m_features[1,0] += f[1]
    #     m_features[2,0] += f[2]
    #     m_features[3,0] += f[3]
    # # Normalize
    # for q in range(4):
    #     m_features[q,0] /= len(features)
    m_features[:, 0] = np.mean(features, axis=0)

    # If using rolling mean, compute with passdown mean
    if classifier_type == 'kernel_mean' and len(past_m_features) > 1:
        # mean decays wrt mean_alpha
        m_features[0, 0] += mean_alpha * past_m_features[0]
        m_features[1, 0] += mean_alpha * past_m_features[1]
        m_features[2, 0] += mean_alpha * past_m_features[2]
        m_features[3, 0] += mean_alpha * past_m_features[3]

        for q in range(4):
            m_features[q, 0] /= (1 + mean_alpha)

    # Compute variances
    # for f in features:
    #     m_features[0, 1] += (f[0] - m_features[0,0])**2
    #     m_features[1, 1] += (f[1] - m_features[1,0])**2
    #     m_features[2, 1] += (f[2] - m_features[2,0])**2
    #     m_features[3, 1] += (f[3] - m_features[3,0])**2
    # # Normalize
    # for q in range(4):
    #     m_features[q, 1] /= len(features)
    m_features[:, 1] = np.var(features, axis=0)

    i = 0
    n_deb = 0
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        if classify(m_features, features[i], classifier):
            cv2.rectangle(c_frame, (x-10,y-10), (x+10+w, y+10+h), (255, 0, 0), 2)
            n_deb += 1
        i += 1

    # If we write out to video
    if w_vid:
        merge_frame = cv2.cvtColor(c_frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(merge_frame)

    # cv2.imshow('Frame', c_frame)
    # cv2.imwrite("media/im_with_keypoints_n_5_mean_thresh.png", c_frame)

    # Return frame stats for performance analysis
    return [0, len(features), n_deb], m_features[:, 0]

# Takes in mean feature vectors and computes distance from the mean
# Returns 1 if far enough away, 0 if not
# Third parameter is classification type

def classify(m_features, feature, classifier):
    # Square error with fixed margin
    if classifier_type == 'naive':
        # Thresh
        # Squared distance
        err = 0
        err_thresh = 0
        # Iterate through features, calculate the error variance
        for i in range(len(feature)):
            err += (m_features[i, 0] - feature[i]) ** 2
            err_thresh += m_features[i, 1]

        # print(err)

        # If error is greater than the mean variance, then debris
        if err > 1.5*err_thresh:
            return True

    # Gaussian probability, use mean and variance too
    elif classifier_type == 'gmm' or 'kmeans':
        _feature = np.array(feature)
        pred = classifier.predict(_feature.reshape(1, -1))
        return pred


def extract(contours):
    features = []

    # Compute bounding box and draw it on the frame
    for c in contours:
        # Pixel area of contour
        area = cv2.contourArea(c)
        # Perimeters of contour
        perimeter = cv2.arcLength(c, True)
        # How close the shape fits a circle
        circularity = 4*np.pi / (perimeter**2)
        # Area of min enclosing circle
        _, radius = cv2.minEnclosingCircle(c)
        min_cir_area = np.pi * radius ** 2

        features.append([area, perimeter, circularity, min_cir_area])
    return features

def kFold(k, gt_stats):

    classifier = None

    if classifier_type == 'gmm':
        classifier = GaussianMixture(n_components=2, max_iter=10000, covariance_type='full')
    elif classifier_type == 'kmeans':
        classifier = KMeans(n_clusters=2, max_iter=10000)

    features = None
    with open('./pickle/' + fname[:-3] + '_' + fname[-2:] + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
    features = np.array(features)
    N = len(features)
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
        print('#####################')
        print(i)
        print(np.sum(result))
        print('#####################')



def main():
    vr = cv2.VideoCapture(fpath)

    # Control whether we write video or not
    debug = True
    write_video = False
    # if write_video:
    print("Writing result out to: " + fout_name)
    vw = cv2.VideoWriter(fout_name, cv2.VideoWriter_fourcc(*'MPEG'), 60, (1920, 1080))

    if not vr.isOpened():
        raise Exception("Error opening video stream or file")

    # Track which frame index we are currently on
    i = 0

    # Create array for fused frames
    f_frames = np.zeros((num_frames, 1080, 1920))

    # Track classification statistics to evaluate performance
    class_stats = []
    # Read in gt
    gt_stats = read_gt(gt_fpath)
    m_features = []

    # kFold(10, gt_stats)

    while vr.isOpened():
        if debug and i % 10 == 0:
            print("On frame: " + str(i))
            if i > 30:
                break

        c_frame = np.zeros((1080, 1920))
        # Store most recent frames

        # Treat f_frames as a circular array
        ret, frame_in = vr.read()
        i += 1

        # If we succesfully read in frame, process it through pipeline
        if ret:
            f_frames[i % num_frames] = cv2.cvtColor(frame_in, cv2.COLOR_RGB2GRAY)
            ret, f_frames[i % num_frames] = cv2.threshold(f_frames[i % num_frames], 1, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), 'uint8')
            f_frames[i % num_frames] = cv2.dilate(f_frames[i % num_frames], kernel, iterations=1)

            if i % frame_stride == 0:
                # Combine current frame buffer
                for j in range(num_frames):
                    c_frame = np.add(c_frame, f_frames[j])

                # Rescale image to be in bounds after summation
                c_frame = cv2.convertScaleAbs(c_frame)

                # Extract contours from fused image
                contours, hierarchy = cv2.findContours(c_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Extract features from contours
                features = extract(contours)

                # Classify and draw boxes around identified outliers
                frame_stats, m_features = drawBoundingBoxes(contours, c_frame, features, write_video, vw, m_features)
                frame_stats[0] = i

                class_stats.append(frame_stats)
                # if(init_f):
                #     class_stats = frame_stats
                #     init_f = 0
                # else:
                #     class_stats = np.append(class_stats, frame_stats, axis=1)

        # Freeze until any key pressed. Quit on pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vr.release()
    vw.release()
    cv2.destroyAllWindows()

    # Plot classification performance against groundtruth
    print(class_stats)
    print(gt_stats)

    class_stats = np.array(class_stats)
    plot_stats(class_stats, gt_stats)

    print("Completed")

if __name__ == '__main__':
    main()