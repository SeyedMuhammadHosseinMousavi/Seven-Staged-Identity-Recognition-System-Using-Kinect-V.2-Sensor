%% Gender Detection
% Created by: Seyed Muhammad Hossein Mousavi
% mosavi.a.i.buali@gmail.com
% This is demo of :
% Mousavi, Seyed Muhammad Hossein, and Atiye Ilanloo. "Seven Staged Identity Recognition System Using Kinect V. 2 Sensor." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.

clc;
clear;
close all;
warning('off');

% Reading Faces
path='Face';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);end;

% Edge Detection
for i = 1 : filesnumber(1,1)
edges{i} = sharppolished(images{i});
disp(['Edge Detected :   ' num2str(i) ]);end;
% imshow(edges{7});

% Extract LPQ Features 
% More value for winsize, better result
winsize=19;
for i = 1 : filesnumber(1,1)
tmp{i}=lpq(edges{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
LPQ(i,:)=tmp{i};end;

% Extract LBP Features
for i = 1 : filesnumber(1,1)
% The less cell size the more accuracy 
lbp{i} = extractLBPFeatures(edges{i},'CellSize',[128 128]);
disp(['Extract LBP :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
lbpfeature(i,:)=lbp{i};
disp(['LBP To Matrix :   ' num2str(i) ]);
end;
FinalReady=[lbpfeature LPQ];

% Labeling for Supervised Learning
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
% FinalReady(1:80,sizefinal+1)=1;
% FinalReady(81:120,sizefinal+1)=2;

%% Lasso Feature Extraction (dimensionality reduction)
disp(['Working On Lasso For Gabor (Please Wait) ...']);
clear lasso;clear B;clear Stats;clear ds;
label(1:80,1)=1;
label(81:120,1)=2;
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

%% Labeling for Classification
% clear test;
% test=FinalReady;
sizefinal=size(test);
sizefinal=sizefinal(1,2);
test(1:80,sizefinal+1)=1;
test(81:120,sizefinal+1)=2;

% Inputs and Targets
Inputs=test(:,1:end-1)';
Targets=test(:,end)';
data.Inputs=Inputs;
data.Targets=Targets;
Inputs=data.Inputs';
Targets=data.Targets';
Targets=Targets(:,1);
nSample=size(Inputs,1);
% Shuffle Data
S=randperm(nSample);
Inputs=Inputs(S,:);
Targets=Targets(S,:);
% Train Data
pTrain=0.7;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;

%% Generate Basic FIS
fis=CreateInitialFIS(data,3);

% Train Data
TrainOutputs=evalfis(data.TrainInputs,fis);
PlotResults(data.TrainTargets,TrainOutputs,'Train Data');
% Test Data
TestOutputs=evalfis(data.TestInputs,fis);
PlotResults(data.TestTargets,TestOutputs,'Test Data');
TrainError= mae(sum(TrainTargets)-sum(TrainOutputs))
TestError= mae(sum(TestTargets)-sum(TestOutputs))
TrainAcuracy=100-TrainError
TestAcuracy=100-TestError

%% Tarin Using DE
fis=TrainUsingDE(fis,data);
