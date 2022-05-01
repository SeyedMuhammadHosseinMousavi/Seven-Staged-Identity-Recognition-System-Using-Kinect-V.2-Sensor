%% Voice Recognition 
% Created by: Seyed Muhammad Hossein Mousavi
% mosavi.a.i.buali@gmail.com
% This is demo of :
% Mousavi, Seyed Muhammad Hossein, and Atiye Ilanloo. "Seven Staged Identity Recognition System Using Kinect V. 2 Sensor." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.

clc;
clear;
close all;
warning('off');

%% Reading Signals
path='voice';
fileinfo = dir(fullfile(path,'*.mp3')); 
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
[signal{i} fs{i}] = audioread(fullfile(path,fileinfo(i).name));
disp(['Loading Sound No :   ' num2str(i) ]);end;

% Noise removal by median filter
for i=1 : filesnumber(1,1)
SmoothMedian{i}=medfilt1(signal{i},20);
disp(['Noise Removal Median :   ' num2str(i) ]);
end;
for i=1 : filesnumber(1,1)
s(:,i)=SmoothMedian{i}(1:96240,1);
disp(['Cell to Array :   ' num2str(i) ]);end;
plot(signal{9}); title('original and smooth');
hold on;
plot(SmoothMedian{9});
fs2=fs{1};
fs3=44100;
sizes=size(s);
sizes=sizes(1,1);

%% MFCC feature extraction
% Define variables
Tw = 100;              % analysis frame duration (ms)
Ts = 10;               % analysis frame shift (ms)
alpha = 0.97;          % preemphasis coefficient
M = 50;                % number of filterbank channels 
C = 0;                  % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 300;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)
for i = 1 : filesnumber(1,1)
MFCC(i,:)=mfcc(s(1:sizes,i),fs2,Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L ); 
disp(['MFCC No :   ' num2str(i) ]);end;
sizemfcc=size(MFCC);
MFCC(MFCC == -inf) = 1;

%% Label
FinalReady=MFCC;
FinalReady(1:6,sizemfcc(1,2))=1;
FinalReady(7:12,sizemfcc(1,2))=2;
FinalReady(12:18,sizemfcc(1,2))=3;

%% SVM Classification
lblknn=FinalReady(:,end);
dataknn=FinalReady(:,1:end-1);
svmclass = fitcecoc(dataknn,lblknn);
svmerror = resubLoss(svmclass);
CVMdl = crossval(svmclass);
genError = kfoldLoss(CVMdl);
% Compute validation accuracy
SVMAccuracy = 1 - kfoldLoss(CVMdl, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedsvm = resubPredict(svmclass);
sizenet=size(FinalReady);
sizenet=sizenet(1,1);
ct=0;
for i = 1 : sizenet(1,1)
if lblknn(i) ~= predictedsvm(i)
ct=ct+1;
end;
end;
% Compute Accuracy
finsvm=ct*100/ sizenet;
SVMAccuracy=(100-finsvm);
% Plot Confusion Matrix
figure
cmsvm = confusionchart(lblknn,predictedsvm);
cmsvm.Title = ['SVM Classification =  ' num2str(SVMAccuracy) '%'];
cmsvm.RowSummary = 'row-normalized';
cmsvm.ColumnSummary = 'column-normalized';






