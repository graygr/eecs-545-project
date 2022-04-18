# Read in .mp4 file, run openCV to binarize, dilate stars, fuse frames together
# then run blob detection from openCV to extract location, area, and shape. 

import cv2
import numpy as np

num_frames = 3
fpath = "AMOS2019-master/assets/data/simple-bg.mp4"

def combineFrames(vr):
    c_frame = np.zeros((1080,1920))
    for i in range(num_frames):
        ret, frame = vr.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            ret, frame = cv2.threshold(frame, 1,255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), 'uint8')
            frame = cv2.dilate(frame, kernel, iterations=1)
        if i == 1:
            c_frame = frame.copy()
            continue
        else:
            c_frame = np.add(c_frame, frame)
    return c_frame

def avgContoursArea(contours):
    mean_area = 0
    # Find average area
    for c in contours:
        mean_area += cv2.contourArea(c)
    mean_area = mean_area / len(contours)
    return mean_area

def drawBBox(contours, c_frame):
    # Compute bounding box and draw it on the frame
    mean_area = avgContoursArea(contours)
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        perimeter = cv2.contourArea(c)
        if area > mean_area:
            cv2.rectangle(c_frame, (x-10,y-10), (x+10+w, y+10+h), (255, 0, 0), 2)
    cv2.imshow('Frame', c_frame)
    cv2.imwrite("media/im_with_keypoints_n_3_gtn_mean_area.png", c_frame)

def extract(contours, c_frame):
    features = []
    bboxes = []
    # Compute bounding box and draw it on the frame
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4*np.pi / (perimeter**2)
        features.append([area, perimeter, circularity])
        bboxes.append([x,y,w,h])
    return features, bboxes


def main():
    vr = cv2.VideoCapture(fpath)
    if not vr.isOpened():
        raise Exception("Error opening video stream or file")
    while(vr.isOpened()):
        c_frame = combineFrames(vr)
        contours, hierarchy = cv2.findContours(c_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        features, bboxes = extract(contours, c_frame)
        print(len(features))
        print(len(bboxes))
        # drawBBox(contours, c_frame)
        # Freeze until any key pressed. Quit on pressing q
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
    vr.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 
