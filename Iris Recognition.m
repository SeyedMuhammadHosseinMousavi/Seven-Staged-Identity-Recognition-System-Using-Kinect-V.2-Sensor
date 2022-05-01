% Iris Recognition
% Created by: Seyed Muhammad Hossein Mousavi
% mosavi.a.i.buali@gmail.com
% This is demo of :
% Mousavi, Seyed Muhammad Hossein, and Atiye Ilanloo. "Seven Staged Identity Recognition System Using Kinect V. 2 Sensor." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.

clc;
clear;
close all;
warning('off');

% Reading Iris Images
path='iris';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);end;

% Gray Level
for i = 1 : filesnumber(1,1)
Gray{i}=rgb2gray(images{i}); 
disp(['To Gray :   ' num2str(i) ]);end;

% Contrast Adjustment
for i = 1 : filesnumber(1,1)
adjusted{i}=imadjust(Gray{i}); 
disp(['Image Adjust :   ' num2str(i) ]);
end;

% Extract LPQ Features 
% More value for winsize, better result
winsize=19;
for i = 1 : filesnumber(1,1)
tmp{i}=lpq(adjusted{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
LPQ(i,:)=tmp{i};end;

% Extract LBP Features
for i = 1 : filesnumber(1,1)
% The less cell size the more accuracy 
lbp{i} = extractLBPFeatures(adjusted{i},'CellSize',[128 128]);
disp(['Extract LBP :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
lbpfeature(i,:)=lbp{i};
disp(['LBP To Matrix :   ' num2str(i) ]);
end;

%% Extract HOG Features
for i = 1 : filesnumber(1,1)
% The less cell size the more accuracy 
hog{i} = extractHOGFeatures(adjusted{i},'CellSize',[64 64]);
disp(['Extract HOG :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
hogfeature(i,:)=hog{i};
disp(['HOG To Matrix :   ' num2str(i) ]);
end;

FinalReady=[lbpfeature LPQ hogfeature];

% Labeling for Supervised Learning
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
% FinalReady(1:80,sizefinal+1)=1;
% FinalReady(81:120,sizefinal+1)=2;

% Lasso Feature Extraction (dimensionality reduction)
disp(['Working On Lasso For Gabor (Please Wait) ...']);
clear lasso;clear B;clear Stats;clear ds;
label(1:10,1)=1;
label(11:20,1)=2;
label(21:30,1)=2;
[B Stats] = lasso(FinalReady,label, 'CV', 5);
disp(B(:,1:5))
disp(Stats)
lassoPlot(B, Stats, 'PlotType', 'CV')
ds.Lasso = B(:,Stats.IndexMinMSE);
disp(ds)
sizemfcc=size(FinalReady);temp=1;       
for i=1:sizemfcc(1,2)
if ds.Lasso(i)~=0
lasso(:,temp)=FinalReady(:,i);temp=temp+1;end;end;
test=lasso;
sizefinal=size(test);
sizefinal=sizefinal(1,2);
test(1:10,sizefinal+1)=1;
test(11:20,sizefinal+1)=2;
test(21:30,sizefinal+1)=3;
%% Shallow Neural Network 
% clear;
netdata=test;
% netdata=netdata.test;
% Labeling For Classification
network=netdata(:,1:end-1);
netlbl=netdata(:,end);
sizenet=size(network);
sizenet=sizenet(1,1);
for i=1 : sizenet
if netlbl(i) == 1
netlbl2(i,1)=1;
elseif netlbl(i) == 2
netlbl2(i,2)=1; 
elseif netlbl(i) == 3
netlbl2(i,3)=1;
end
end
% Changing data shape from rows to columns
network=network'; 
% Changing data shape from rows to columns
netlbl2=netlbl2'; 
% Defining input and target variables
inputs = network;
targets = netlbl2;
% Create a Pattern Recognition Network
hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
% Polak-Ribiére Conjugate Gradient
net = feedforwardnet(20, 'traincgp');
%
[net,tr] = train(net,inputs,targets);
% Test the Network
outputs = net(inputs);
%
errors = gsubtract(targets,outputs);
%
performance = perform(net,targets,outputs)
% Polak-Ribiére Conjugate Gradient
figure, plottrainstate(tr)
% Plot Confusion Matrixes
figure, plotconfusion(targets,outputs);
title('Polak-Ribiére Conjugate Gradient');

