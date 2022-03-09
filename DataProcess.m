% File Created 3/9/2022
% Purpose: Read video file, preprocess it
% - Pass binary classification filter over it
% - Dilate pixels

% For Gregor's computer, comment out if diff
cd("C:\Users\akgre\Documents\Masters\EECS 545\eecs-545-project")

% Load Video
vr = VideoReader("AMOS2019-master/assets/data/simple-fg.mp4")

% Store first frame
frame = readFrame(vr)
imshow(frame)

% Binary Filter

% Dilate Pixels

