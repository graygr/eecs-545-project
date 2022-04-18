from features import *
from sklearn.mixture import GaussianMixture

fpath = "AMOS2019-master/assets/data/simple-bg.mp4"

def main():
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
            features = extract(contours)

            classifier = GaussianMixture(n_components=2, max_iter=10000, covariance_type='full')
            classifier.fit(features)

            y_hat = classifier.predict(features)
            print(y_hat)
            raise

        # drawBBox(contours, c_frame)
        # Freeze until any key pressed. Quit on pressing q
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
    vr.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 

