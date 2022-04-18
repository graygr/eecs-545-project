from features import *
from sklearn.mixture import GaussianMixture

fpath = "AMOS2019-master/assets/data/simple-bg.mp4"

def main():
    vr = cv2.VideoCapture(fpath)
    if not vr.isOpened():
        raise Exception("Error opening video stream or file")
    while(vr.isOpened()):
        c_frame = combineFrames(vr)
        contours, hierarchy = cv2.findContours(c_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        features, bboxes = extract(contours, c_frame)

        classifier = GaussianMixture(n_components=2, max_iter=1000, covariance_type='full')
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
