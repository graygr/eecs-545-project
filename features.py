# Read in .mp4 file, run openCV to binarize, dilate stars, fuse frames together
# then run blob detection from openCV to extract location, area, and shape. 

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from kfold import kFold

num_frames = 10
frame_stride = 3
kernel_size = 5
fname = "simple-bg"
classifier_type = 'gmm'
mean_alpha = 0.9
sensitivity = 10
# 10

fpath = "AMOS2019-master/assets/data/" + fname + ".mp4"
fout_name = fname + '-' + classifier_type + ".avi"

# GT file paths
gt_fpath = "groundtruth-files/" + fname + "-gt.csv"

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
        if j < len(gt_stats):
            frame_index = int(frame[0])
            # If current frame is greater than the window, then bump up the next
            if frame_index > int(gt_stats[j][0]):
                # print("i: " + str(i))
                # print("j: " + str(j))
                # print("gt_frame: " + str(int(gt_stats[j][0])))
                j += 1
            if j < len(gt_stats):
                gt_interpolated.append(gt_stats[j].copy())
                gt_interpolated[i - 1][0] = frame_index

            i += 1

    # Plot
    gt_interpolated = np.array(gt_interpolated)

    # Write out for later
    np.savetxt(fname + "-" + classifier_type + "-class_stats.csv", class_stats, delimiter=",")
    np.savetxt(fname + "-" + classifier_type + "-gt_interp.csv", gt_interpolated, delimiter=",")

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

    m_features = np.zeros((len(features[0]),2))

    # Compute means
    m_features[:, 0] = np.mean(features, axis=0)

    # If using rolling mean, compute with passdown mean
    if classifier_type == 'kernel_mean' and len(past_m_features) > 1:
        # mean decays wrt mean_alpha
        for q in range(len(features[0])):
            m_features[q, 0] += mean_alpha * past_m_features[q]
            m_features[q, 0] /= (1 + mean_alpha)

    # Wrap angle to pi
    if (m_features[3, 0] > 180):
        m_features[3, 0] = m_features[3, 0] % 180

    # Compute variances
    m_features[:, 1] = np.std(features, axis=0)

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
            # print(feature[i])
            # Normalize to feature magnitude
            err += ((m_features[i, 0] - feature[i]) / m_features[i, 0]) ** 2
            # err_thresh += m_features[i, 1]

        # print(m_features[:, 1])
        # print(str(err) + " " + str(err_thresh))

        # If error is greater than the mean variance, then debris
        if err > sensitivity:
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
        # print(len(c))
        # Pixel area of contour
        area = cv2.contourArea(c)
        # Perimeters of contour
        perimeter = cv2.arcLength(c, True)
        # Aspect ratio
        _,_,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        # Radius of min enclosing circle
        _, radius = cv2.minEnclosingCircle(c)
        radius = radius ** 2
        # Orientation of the object
        if(len(c) < 5):
            orientation = 0
        else:
            _, _, orientation = cv2.fitEllipseAMS(c)
        # # print(orientation)

        features.append([perimeter, area, radius, orientation, aspect_ratio])
    return features

def write_csv(class_stats):
    import pandas as pd
    pd.DataFrame(class_stats).iloc[::3].to_csv(
        './class-stats/' + fname + '-' + classifier_type + '-class_stats_newfeature.csv',
        sep='\t', index=False,  header=None)


def main():
    # class_stats = kFold(10, classifier_type, fname)
    # write_csv(class_stats)
    # raise
    vr = cv2.VideoCapture(fpath)

    # # Control whether we write video or not
    debug = True
    write_video = False
    # if write_video:
    if write_video:
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


    start_time = time.time()

    while vr.isOpened():
        if debug and i % 10 == 0:
            print("On frame: " + str(i))
            if i > 4000:
                break


        c_frame = np.zeros((1080, 1920))
        # Store most recent frames

        # Treat f_frames as a circular array
        ret, frame_in = vr.read()

        # If we succesfully read in frame, process it through pipeline
        if ret:
            f_frames[i % num_frames] = cv2.cvtColor(frame_in, cv2.COLOR_RGB2GRAY)
            ret, f_frames[i % num_frames] = cv2.threshold(f_frames[i % num_frames], 50, 255, cv2.THRESH_BINARY)
            kernel = np.ones((kernel_size, kernel_size), 'uint8')
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
                # cv2.imshow('Frame', c_frame)
                # Show image
        else:
            break
        # Freeze until any key pressed. Quit on pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    end_time = time.time()
    print("Total runtime: " + str(end_time - start_time))
    print("Average time per frame: " + str((end_time - start_time) / i))

    vr.release()
    vw.release()
    cv2.destroyAllWindows()

    # Plot classification performance against groundtruth
    # class_stats = kFold(10, classifier_type, fname)

    # Write class stats out to CSV

    class_stats = np.array(class_stats)
    plot_stats(class_stats, gt_stats)

    print("Completed")

if __name__ == '__main__':
    main()