clc; clear all; close all;

%% load video, add center fixation cross
vid = VideoReader('gaze_center_test.mp4');
outputVideo = VideoWriter('gaze_center_wfixation.mp4'); 
outputVideo.FrameRate = vid.FrameRate;
open(outputVideo);

% Process and write each frame
while hasFrame(vid)
    frame = readFrame(vid);
    frameSize = size(frame);
    centerX = frameSize(2) / 2;
    centerY = frameSize(1) / 2;

    crossSize = 20; 
    lineWidth = 3;  
    color = [0, 0, 0]; 
    frame = insertShape(frame, 'Line', [centerX, centerY - crossSize, centerX, centerY + crossSize, ...
                                        centerX - crossSize, centerY, centerX + crossSize, centerY], ...
                                        'Color', color, 'LineWidth', lineWidth);
    writeVideo(outputVideo, frame);
end

close(outputVideo);

vid = VideoReader('gaze_center_wfixation.mp4.avi');

%% extract frames during a fixation
startSecond = 1; 
endSecond = 5; 
stepSize = 10; 
nFrames = floor((endSecond-startSecond)*1000/stepSize/vid.FrameRate);

startFrame = round(startSecond * vid.FrameRate);
endFrame = round(endSecond * vid.FrameRate);
currentFrame = 1; frameCounter = 1;

while hasFrame(vid) && currentFrame <= endFrame
    frame = readFrame(vid);
    if currentFrame >= startFrame && mod(currentFrame - startFrame, stepSize) == 0
        fileName = sprintf('%d.jpg', frameCounter); 
        fullPath = fullfile('extractedFrames/gaze_center_fixation_test', fileName);
        imwrite(frame, fullPath); 
        frameCounter = frameCounter + 1;
    end
    currentFrame = currentFrame + 1;
end

%% load images
frameCell = cell(nFrames, 1); 
for i = 1:nFrames
    fileName = sprintf('%d.jpg', i);
    fullPath = fullfile('extractedFrames/gaze_center_fixation_test',fileName);
    frameCell{i} = double(imread(fullPath))/255;
end

%% compute curvature
% get pixel intensity values for each frame
for i = 1:nFrames
    frameCell{i} = 0.2989*frameCell{i}(:,:,1)...
                 + 0.5879*frameCell{i}(:,:,2)...
                 + 0.1140*frameCell{i}(:,:,3);
end

% convert to mat
for i = 1:nFrames
    frameMat(:,:,i) = frameCell{i};
end

% pixel intensity
pixel_int = reshape(frameMat, [], nFrames);

% get the difference vector
unit = @(v) v/norm(v); vec_t = NaN(length(pixel_int),nFrames-1);
for i = 1:nFrames-1
    vec_t(:,i) = unit(pixel_int(:,i+1) - pixel_int(:,i));
end

% calculate curvature
c_t = NaN(nFrames-2,1);
for i = 1:nFrames-2
    c_t(i) = acos(vec_t(:,i)'*vec_t(:,i+1));
end
global_curv = mean(rad2deg(c_t));


%% show video
for i = 1:nFrames
    imagesc(frameMat(:,:,i)); colormap('gray'); axis image; box off, axis off;
    pause(.1)
end
