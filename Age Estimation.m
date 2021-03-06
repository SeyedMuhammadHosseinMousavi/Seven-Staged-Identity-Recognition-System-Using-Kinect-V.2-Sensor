%% Age Estimation
% Created by: Seyed Muhammad Hossein Mousavi
% mosavi.a.i.buali@gmail.com
% This is demo of :
% Mousavi, Seyed Muhammad Hossein, and Atiye Ilanloo. "Seven Staged Identity Recognition System Using Kinect V. 2 Sensor." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.

clc;clear;
%% Proposed face extraction
% Reading depth image
I1 = imread('d1.jpg');
% Image resizing
Res = imresize(I1,[256 256]);
% Converting to gray level
I2 = rgb2gray(Res);
Res1 = I2;
% Removing black pixels
size1=size(Res1);
size1=size1(1,2);
for i=1:size1 % Loop for row navigation
    for j=1:size1 % Loop for column navigation
    if Res1(i,j)>235 || Res1(i,j)<70 % Conditional statment for black spots threshold
        Res1(i,j)=255; % Replacing value for black spots
    else
        Res1(i,j)=Res1(i,j); % Do not effect any change
    end
    end
end
% Local standard deviation of image
J = rangefilt(Res1);
% Ellipse fitting
BW=imbinarize(Res1);;
BW = imclearborder(Res1); % get rid of objects connected to the image boundary
% BW = bwareafilt(BW,1); % pick the single largest object 
BW = bwconvhull(BW); % after using bwareafilt
% Calculate centroid, orientation and major/minor axis length of the ellipse
s = regionprops(BW,{'Centroid','Orientation','MajorAxisLength','MinorAxisLength'});
% Calculate the ellipse line
theta = linspace(0,2*pi);
col = (s.MajorAxisLength/2.5)*cos(theta);
row = (s.MinorAxisLength/2.5)*sin(theta);
M = makehgtform('translate',[s.Centroid, 0],'zrotate',deg2rad(-1*s.Orientation));
D = M*[col;row;zeros(1,numel(row));ones(1,numel(row))];
%# Create an ellipse shaped mask
c = fix(size(I2) / 2);   %# Ellipse center point (y, x)
r_sq = [78, 103] .^ 2;  %# Ellipse radii squared (y-axis, x-axis)
[X, Y] = meshgrid(1:size(I2, 2), 1:size(I2, 1));
ellipse_mask = (r_sq(2) * (X - c(2)) .^ 2 + ...
    r_sq(1) * (Y - c(1)) .^ 2 <= prod(r_sq));
%# Apply the mask to the image
A_cropped = bsxfun(@times, I2, uint8(ellipse_mask));

for i=1:size1 % Loop for row navigation
    for j=1:size1 % Loop for column navigation
    if A_cropped(i,j)== 0 
        A_cropped(i,j)= 255; % Replacing value for black spots
    else
        A_cropped(i,j)= A_cropped(i,j); % Do not effect any change
    end
    end
end
% Removing Extra
ext = imcrop(A_cropped,[64 64 125 130]);
% Plotting Results
subplot(2,3,1)
subimage(I2);title('Gray Level');
subplot(2,3,2)
subimage(J);title('Standard Deviation');
subplot(2,3,3)
imshow(J)
hold on
plot(D(1,:),D(2,:),'r','LineWidth',2);title('Ellipse Fitting');
subplot(2,3,4)
imshow(I2)
hold on
plot(D(1,:),D(2,:),'r','LineWidth',2);title('Gray Ellipse Fitting');
subplot(2,3,5)
imshow(A_cropped);;title('Ellipse Crop');
subplot(2,3,6)
subimage(ext);title('No Extra');

% Second image
% Reading 
cI1 = imread('c1.jpg');
% Image resizing
Res = imresize(cI1,[256 256]);
% Converting to gray level
cI2 = rgb2gray(Res);
cRes1 = cI2;
% Removing black pixels
size1=size(Res1);
size1=size1(1,2);
for i=1:size1 % Loop for row navigation
    for j=1:size1 % Loop for column navigation
    if Res1(i,j)>235 || Res1(i,j)<70 % Conditional statment for black spots threshold
        Res1(i,j)=255; % Replacing value for black spots
    else
        Res1(i,j)=Res1(i,j); % Do not effect any change
    end
    end
end
% Local standard deviation of image
cJ = rangefilt(cRes1);
% Ellipse fitting
BW=imbinarize(cRes1);;
BW = imclearborder(cRes1); % get rid of objects connected to the image boundary
% BW = bwareafilt(BW,1); % pick the single largest object 
BW = bwconvhull(BW); % after using bwareafilt
% Calculate centroid, orientation and major/minor axis length of the ellipse
s = regionprops(BW,{'Centroid','Orientation','MajorAxisLength','MinorAxisLength'});
% Calculate the ellipse line
theta = linspace(0,2*pi);
col = (s.MajorAxisLength/2.5)*cos(theta);
row = (s.MinorAxisLength/2.5)*sin(theta);
M = makehgtform('translate',[s.Centroid, 0],'zrotate',deg2rad(-1*s.Orientation));
D = M*[col;row;zeros(1,numel(row));ones(1,numel(row))];
%# Create an ellipse shaped mask
c = fix(size(cI2) / 2);   %# Ellipse center point (y, x)
r_sq = [78, 103] .^ 2;  %# Ellipse radii squared (y-axis, x-axis)
[X, Y] = meshgrid(1:size(cI2, 2), 1:size(cI2, 1));
ellipse_mask = (r_sq(2) * (X - c(2)) .^ 2 + ...
    r_sq(1) * (Y - c(1)) .^ 2 <= prod(r_sq));
%# Apply the mask to the image
cA_cropped = bsxfun(@times, cI2, uint8(ellipse_mask));

for i=1:size1 % Loop for row navigation
    for j=1:size1 % Loop for column navigation
    if cA_cropped(i,j)== 0 
        cA_cropped(i,j)= 255; % Replacing value for black spots
    else
        cA_cropped(i,j)= cA_cropped(i,j); % Do not effect any change
    end
    end
end
% Removing Extra
cext = imcrop(cA_cropped,[64 64 125 130]);
% Plotting Results
figure
subplot(2,3,1)
subimage(cI2);title('Gray Level');
subplot(2,3,2)
subimage(cJ);title('Standard Deviation');
subplot(2,3,3)
imshow(cJ)
hold on
plot(D(1,:),D(2,:),'r','LineWidth',2);title('Ellipse Fitting');
subplot(2,3,4)
imshow(cI2)
hold on
plot(D(1,:),D(2,:),'r','LineWidth',2);title('Gray Ellipse Fitting');
subplot(2,3,5)
imshow(cA_cropped);;title('Ellipse Crop');
subplot(2,3,6)
subimage(cext);title('No Extra');
%% Proposed age estimation
% Selecting nose
nose = imcrop(cext,[30 65 75 70]);
% subimage(nose);
nosed = imcrop(ext,[30 65 75 70]);
% subimage(nosed);
% Calculating image entropy
ec = entropyfilt(nose);
% imshow(ec,[]);
ed = entropyfilt(nosed);
% imshow(ed,[]);
% Calculating pixel sum
dsum=sum(sum(ed));
csum=sum(sum(ec));
finsum=dsum+csum;
% Rounding number
y = round(finsum);
% Divide by bigest age in dataset
divfin=y/70;
% Divide by smallest age in dataset by 2 
lowestage= 9*2;
finalage=divfin/lowestage;
disp(['Age is :   ' num2str(round(finalage)) ]);
% Plots
figure
subplot(2,2,1)
imshow(nose);title('Nose color');
subplot(2,2,2)
imshow(nosed);title('Nose depth');
subplot(2,2,3)
imshow(ec,[]);title('entropy color');
subplot(2,2,4)
imshow(ed,[]);title('entropy depth');