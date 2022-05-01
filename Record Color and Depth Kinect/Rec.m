clear;
% Add Utility Function to the MATLAB Path
utilpath = fullfile(matlabroot, 'toolbox', 'imaq', 'imaqdemos','html', 'KinectForWindows');
addpath(utilpath);
% separate VIDEOINPUT object needs to be created for each of the color and depth(IR) devices
% The Kinect for Windows Sensor shows up as two separate devices in IMAQHWINFO.
hwInfo = imaqhwinfo('kinect')
hwInfo.DeviceInfo(1)
hwInfo.DeviceInfo(2)
% Create the VIDEOINPUT objects for the two streams
colorVid = videoinput('kinect',1)
depthVid = videoinput('kinect',2)
preview(colorVid);
preview(depthVid);
% Set the triggering mode to 'manual'
triggerconfig([colorVid depthVid],'manual');
%  In this example 100 frames are acquired to give the Kinect for Windows sensor sufficient time to
%  start tracking a skeleton.
numberofframe=160;
colorVid.FramesPerTrigger = numberofframe;
depthVid.FramesPerTrigger = numberofframe;
% Start the color and depth device. This begins acquisition, but does not
% start logging of acquired data.
start([colorVid depthVid]);
% Trigger the devices to start logging of data.
trigger([colorVid depthVid]);
% Retrieve the acquired data
[colorFrameData,colorTimeData,colorMetaData] = getdata(colorVid);
[depthFrameData,depthTimeData,depthMetaData] = getdata(depthVid);
% Stop the devices
stop([colorVid depthVid]);

%% converting 4-d matrix to 3-d rgb images
rgb4=size(colorFrameData)
for i=1:rgb4(1,4)
    RGB{i}=colorFrameData(:,:,:,i)
end;
%%%%%%%%%%%%%%%%%%%%%%%%%

% converting 4-d matrix to 3-d depth images
depth4=size(depthFrameData)
for i=1:depth4(1,4)
    Depth{i}=depthFrameData(:,:,:,i)
end;

%% Saving image to drive c(RGB)
%first delete previews files from specific folder 
delete('c:\My Database RGB and Depth\H\RGB Frames\*.jpg');
%then saving new files
for i = 1 : rgb4(1,4)
imwrite(RGB{i},['c:\My Database RGB and Depth\H\RGB Frames\H RGB Frame' num2str(i) '.jpg']);
    disp(['No of saved RGB image :   ' num2str(i) ]);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%

%% Saving image to drive c(DEPTH)
%first delete previews files from specific folder 
delete('c:\My Database RGB and Depth\H\Depth Frames\*.png');
%then saving new files
for i = 1 : depth4(1,4)
imwrite(Depth{i},['c:\My Database RGB and Depth\H\Depth Frames\H Depth Frame' num2str(i) '.png']);
    disp(['No of saved DEPTH image :   ' num2str(i) ]);
end;

