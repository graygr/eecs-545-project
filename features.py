# Read in .mp4 file, run openCV to binarize, dilate stars, fuse frames together
# then run blob detection from openCV to extract location, area, and shape. 

import cv2
import numpy as np

num_frames = 10
fpath = "AMOS2019-master/assets/data/simple-bg.mp4"

# Helper function to create a fused frame
# Combines frames from index in back n_frames
# Returns a fused frame from starting index, going back n_frames


# Helper function to compute the average area of all detected contours in the image
def avgContoursArea(contours):
    mean_area = 0
    # Find average area
    for c in contours:
        mean_area += cv2.contourArea(c)
    mean_area = mean_area / len(contours)
    return mean_area

# Draw classification boxes around outliers
def drawBoundingBoxes(contours, c_frame, features):
    # Compute mean features
    # mean_area = avgContoursArea(contours)
    m_features = np.zeros((4,2))

    # Compute means
    for f in features:
        m_features[0,0] += f[0]
        m_features[1,0] = f[1]
        m_features[2,0] = f[2]
        m_features[3,0] = f[3]
    # Normalize
    for q in range(4):
        m_features[q,0] /= len(features)

    # Compute variances
    for f in features:
        m_features[0, 1] += (f[0] - m_features[0,0])**2
        m_features[1, 1] += (f[1] - m_features[1,0])**2
        m_features[2, 1] += (f[2] - m_features[2,0])**2
        m_features[3, 1] += (f[3] - m_features[3,0])**2
    # Normalize
    for q in range(4):
        m_features[q, 1] /= len(features)

    i = 0
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        if classify(m_features, features[i], 0):
            cv2.rectangle(c_frame, (x-10,y-10), (x+10+w, y+10+h), (255, 0, 0), 2)
        i += 1
    cv2.imshow('Frame', c_frame)
    cv2.imwrite("media/im_with_keypoints_n_5_mean_thresh.png", c_frame)



# Takes in mean feature vectors and computes distance from the mean
# Returns 1 if far enough away, 0 if not
# Third parameter is classification type

def classify(m_features, feature, class_mode):
    # Square error
    if class_mode == 0:
        # Thresh
        # TODO: Tune this
        # This thresh works for simple_bg, failes during vibration/movement though
        err_thresh = 230
        # Squared distance
        err = 0
        # Iterate through features
        for i in range(len(feature)):
            err += (m_features[i,0] - feature[i]) ** 2
        err /= len(feature)

        if err > err_thresh:
            return True

    # Gaussian probability, use mean and variance too
    elif class_mode == 1:
        pass


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


def main():
    vr = cv2.VideoCapture(fpath)
    if not vr.isOpened():
        raise Exception("Error opening video stream or file")

    i = 0
    f_frames = np.zeros((num_frames, 1080, 1920))

    while vr.isOpened():
        c_frame = np.zeros((1080, 1920))
        # Store most recent frames


        # Treat f_frames as a circular array
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
            features = extract(contours)

            # Classify and draw boxes around identified outliers
            drawBoundingBoxes(contours, c_frame, features)

        # print(len(features))
        # print(len(bboxes))
        # drawBBox(contours, c_frame)
        # Freeze until any key pressed. Quit on pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vr.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 
