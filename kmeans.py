# File: feature_extraction
# Purpose: Read in .mp4 file, run openCV to binarize, dilate stars, fuse frames together
#          then run blob detection from openCV to extract location, area, and shape. 


from features import *
import matplotlib.pyplot as plt
import timeit
import cv2
from sklearn.cluster import KMeans

def main():
    vr = cv2.VideoCapture(fpath)
    if(not vr.isOpened()):
        raise Exception("Error opening video stream or file")
    i = 0
    f_frames = np.zeros((num_frames, 1080, 1920))
    new_frames = []

    # Loop through video file
    while(vr.isOpened()):

        c_frame = np.zeros((1080,1920))
        ret, frame_in = vr.read()
        i += 1
        print(i)
        
        if ret:
            f_frames[i % num_frames] = cv2.cvtColor(frame_in, cv2.COLOR_RGB2GRAY)
            ret, f_frames[i % num_frames] = cv2.threshold(f_frames[i % num_frames], 1, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), 'uint8')
            f_frames[i % num_frames] = cv2.dilate(f_frames[i % num_frames], kernel, iterations=1)

            # Combine current frame buffer
            c_frame = np.add(c_frame, np.sum(f_frames, axis=0))

            # Rescale image to be in bounds
            c_frame = cv2.convertScaleAbs(c_frame)

            # Extract contours from fused image
            contours, hierarchy = cv2.findContours(c_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.array(contours, dtype=object)

            # Extract features from contours
            features = np.array(extract(contours))

            kmeans = KMeans(n_clusters=2, max_iter=1500).fit(features)

            # identify which class is debris
            # assume that debris have higher mean squared distance from centroid
            class0_indices = np.where(kmeans.labels_ == 0)[0]
            class1_indices = np.where(kmeans.labels_ == 1)[0]

            class0_vec = features[class0_indices, :] - kmeans.cluster_centers_[0]
            class1_vec = features[class1_indices, :] - kmeans.cluster_centers_[1]

            mean_squared_dist0 = np.square(class0_vec).mean()
            mean_squared_dist1 = np.square(class1_vec).mean()

            debris_indices = class0_indices
            if (mean_squared_dist1 > mean_squared_dist0):
                debris_indices = class1_indices

            # draw bounding boxes around debris
            for c in contours[debris_indices]:
                (x,y,w,h) = cv2.boundingRect(c)
                cv2.rectangle(c_frame, (x-10,y-10), (x+10+w, y+10+h), (255, 0, 0), 2)

            # add to final video frames
            new_frames.append(c_frame)

            if i == 60: # skip to a good example and plot some clusters in lower dimensions
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.scatter(features[:, 0], features[:, 1], c=kmeans.labels_, cmap='rainbow', s = 7)
                ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', s = 70, alpha=0.30)
                ax.set_title('simple-bg.mp4, frame 60')
                ax.set_xlabel('area')
                ax.set_ylabel('perimeter')
                fig.show()



        else:
            vr.release()

    # write to video
    frameSize = (1920, 1080)

    out = cv2.VideoWriter('output-simple-bg-5x5-10fr.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize)
    for i in range(len(new_frames)):
        out.write(cv2.cvtColor(new_frames[i], cv2.COLOR_GRAY2BGR))
    out.release()



 

if __name__ == '__main__':
    main() 
