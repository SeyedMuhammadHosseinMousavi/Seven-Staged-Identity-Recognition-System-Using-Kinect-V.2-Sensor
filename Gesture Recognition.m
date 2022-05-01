%% Gesture Recognition
% Created by: Seyed Muhammad Hossein Mousavi
% mosavi.a.i.buali@gmail.com
% This is demo of :
% Mousavi, Seyed Muhammad Hossein, and Atiye Ilanloo. "Seven Staged Identity Recognition System Using Kinect V. 2 Sensor." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.

clc;
clear;
close all;
warning('off');

%% Reading Gestures
path='Gestures';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading image No :   ' num2str(i) ]);end;

%% Gray Level
for i = 1 : filesnumber(1,1)
Gray{i}=rgb2gray(images{i}); 
disp(['To Gray :   ' num2str(i) ]);end;

%% Extract Gabor Features (just depth)
for i = 1 : filesnumber(1,1)
gaborArray = gaborFilterBank(2,2,19,19);  % Generates the Gabor filter bank
featureVector{i} = gaborFeatures(Gray{i},gaborArray,64,64); 
disp(['Extracting Gabor Vector :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
Gaborvector(i,:)=featureVector{i};
disp(['To Matrix :   ' num2str(i) ]);end;

%% Lasso Feature Extraction (dimensionality reduction)
disp(['Working On Lasso For Gabor (Please Wait) ...']);
clear lasso;clear B;clear Stats;clear ds;
label(1:100,1)=1;
label(101:200,1)=2;
label(201:300,1)=3
[B Stats] = lasso(Gaborvector,label, 'CV', 5);
disp(B(:,1:5))
disp(Stats)
lassoPlot(B, Stats, 'PlotType', 'CV')
ds.Lasso = B(:,Stats.IndexMinMSE);
disp(ds)
sizemfcc=size(Gaborvector);temp=1;       
for i=1:sizemfcc(1,2)
if ds.Lasso(i)~=0
lasso(:,temp)=Gaborvector(:,i);temp=temp+1;end;end;
Gabor=lasso;

%% Labeling for Classification
clear test;
test=Gabor;
sizefinal=size(test);
sizefinal=sizefinal(1,2);
test(1:100,sizefinal+1)=1;
test(101:200,sizefinal+1)=2;
test(201:300,sizefinal+1)=3;

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
%% Tarin Using ANFIS
fis=TrainUsingANFIS(fis,data);
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



