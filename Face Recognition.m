%% Face Recognition
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

% Closing Morphology
se = strel('line',2,45);
for i = 1 : filesnumber(1,1)
close{i} = imclose(edges{i},se);
disp(['Closing :   ' num2str(i) ]);end;
% imshow(close{1});

%% Feature Extraction
% Extract SURF Features 
imset = imageSet('Face CNN','recursive'); 
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',20,'PointSelection','Detector');
% Encode the images as new features
SURF = encode(bag,imset);
%-------------------------------------------------
% Extract HOG Features 
for i = 1 : filesnumber(1,1)
% The less cell size the more accuracy 
hog{i} = extractHOGFeatures(close{i},'CellSize',[128 128]);
disp(['Extract HOG :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)
HOG(i,:)=hog{i};
disp(['HOG To Matrix :   ' num2str(i) ]);end;
% Combining Feature Matrixes
FinalReady=[HOG SURF];

% Labeling for Supervised Learning
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
FinalReady(1:40,sizefinal+1)=1;
FinalReady(41:80,sizefinal+1)=2;
FinalReady(81:120,sizefinal+1)=3;

%% SVM Classification
lblknn=FinalReady(:,end);
dataknn=FinalReady(:,1:end-1);
tsvm = templateSVM('KernelFunction','polynomial');
svmclass = fitcecoc(dataknn,lblknn,'Learners',tsvm);
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
% ROC
[~,scoresvm] = resubPredict(svmclass);
diffscoresvm = scoresvm(:,2) - max(scoresvm(:,1),scoresvm(:,3));
[Xsvm,Ysvm,T,~,OPTROCPTsvm,suby,subnames] = perfcurve(lblknn,diffscoresvm,1);
% ROC curve plot
figure;
plot(Xsvm,Ysvm)
hold on
plot(OPTROCPTsvm(1),OPTROCPTsvm(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for SVM')
hold off
