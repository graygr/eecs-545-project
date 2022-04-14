% File Created 3/9/2022
% Purpose: Read video file, preprocess it
% - Pass binary classification filter over it
% - Dilate pixels
clc;clear all;

% For Gregor's computer, comment out if diff
%cd("C:\Users\akgre\Documents\Masters\EECS 545\eecs-545-project")
cd("C:\Users\Gregor Limstrom\Documents\masters\eecs545\eecs-545-project");

video_files = ["AMOS2019-master/assets/data/simple-bg.mp4";
                "AMOS2019-master/assets/data/simple-fg.mp4";
                "AMOS2019-master/assets/data/complex-bg.mp4";
                "AMOS2019-master/assets/data/complex-fg.mp4"
                ];

frame_spacing = [2,5,10,50];
filename = ["sbg_2.csv", "sbg_5.csv", "sbg_10.csv", "sbg_50.csv";
            "sfg_2.csv", "sfg_5.csv", "sfg_10.csv", "sfg_50.csv";
            "cbg_2.csv", "cbg_5.csv", "cbg_10.csv", "cbg_50.csv";
            "cfg_2.csv", "cfg_5.csv", "cfg_10.csv", "cfg_50.csv";
            ];
            

bin_thresh = 0.1;
kern_size = 3;
        
% Iterate through video files
for i = 2:4
    disp("Starting file: ");
    disp(i);
    % Iterate through frame spacing options
    for j = 1:4
        disp("Starting frame option: ");
        disp(j);
        
        % Open video frame stream
        vr = VideoReader(video_files(i));
        
        % Initialize file to write out
        fn = filename(i, j);
        
        % Loop through video combining according to video frames
        for k = 1 : vr.NumFrames/frame_spacing(j)
            disp(k/(vr.NumFrames/frame_spacing(j)));
            
            cf = combine(vr, frame_spacing(j), bin_thresh, kern_size);
            writematrix(cf, fn, "WriteMode", "append");
        end
    end
end
% Load Video
% vr = VideoReader("AMOS2019-master/assets/data/complex-bg.mp4");
% 
% c_f = combine(vr, 5, 0.1, 3);
% imshow(c_f);

function [combined_frame] = combine(videoreader, num_frames, bin_thresh, dilation_factor)
    combined_frame = imdilate(imbinarize(rgb2gray(readFrame(videoreader)), bin_thresh), ones(dilation_factor));
    for i = 1:num_frames - 1
        new_frame = imdilate(imbinarize(rgb2gray(readFrame(videoreader)), bin_thresh), ones(dilation_factor));
        combined_frame = imfuse(combined_frame, new_frame, "blend");
        combined_frame = imbinarize(combined_frame, bin_thresh);
    end
end



