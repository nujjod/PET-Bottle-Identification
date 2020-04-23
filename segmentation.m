% remove after resize
%        imagesfoldername=natsortfiles({'/Users/nujood/Desktop/Bottle-Segmentation/Raw-Images'});
%        maskfoldername=natsortfiles({'/Users/nujood/Desktop/Bottle-Segmentation/Masks'});
%        imds = imageDatastore(imagesfoldername);
%        numTrainingImages = numel(imds.Files); 
%        imds = resizeImages(imds, numTrainingImages);
%        pxds = imageDatastore('/Users/nujood/Desktop/Bottle-Segmentation/Masks');
%        pxds = resizePixelLabels(pxds, numTrainingImages);
% ----------

images_fp = '/Users/nujood/Desktop/Bottle-Segmentation/Images_Updated';
S1 = dir(fullfile(images_fp,'*.png'));
N1 = natsortfiles({S1.name});
F1 = cellfun(@(n)fullfile(images_fp,n),N1,'uni',0);
imds_resized = imageDatastore(F1);

% Specify Classes
classes=["TransparentPET" "BluePET" "Black"];
labelIDs = PixelLabelIDs();

%  create the pixelLabelDatastore
masks_fp = '/Users/nujood/Desktop/Bottle-Segmentation/masks_Updated';
S2 = dir(fullfile(masks_fp,'*.png'));
N2 = natsortfiles({S2.name});
F2 = cellfun(@(n)fullfile(masks_fp,n),N2,'uni',0);
pxds_resized = pixelLabelDatastore(F2, classes, labelIDs);

% Analyze Dataset Statistics
tbl = countEachLabel(pxds_resized);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

%  display one of the pixel-labeled images by overlaying it on top of an image
I = readimage(imds_resized, 1);
C = readimage(pxds_resized, 1);
cmap = ColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B) 
pixelLabelColorbar(cmap,classes);

% Prepare Training, Validation, and Test Sets
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = SplitData(imds_resized, pxds_resized, classes, labelIDs);

% Specify the number of classes.
numClasses = numel(classes);

% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = size(I); %[720 960 3];

% Create DeepLab v3+.
% network = 'resnet50';
network = 'resnet18';
lgraph = deeplabv3plusLayers(imageSize, numClasses, network);

plot(lgraph)

% lgraph.Layers 
% layerGraph(lgraph.Layers)
analyzeNetwork(lgraph)

% imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
% classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name);
 
lgraph = replaceLayer(lgraph,"classification",pxLayer);

% Define validation data.
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.7,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationFrequency', 10,...
    'ValidationPatience', 4); ...

augmenter = imageDataAugmenter('RandXReflection',true,...
     'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

% Start Training
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);

% remove this part after training
% [netDeepLab, info] = trainNetwork(pximds,lgraph,options);
% fprintf('Training is done...\n');
% save('/Users/nujood/Desktop/Bottle-Segmentation/netDeepLab.mat','netDeepLab');

% Test Network
data = load('netDeepLab.mat'); 
net = data.netDeepLab;

% Evaluate Trained Network
pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);
 
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
metrics.DataSetMetrics
metrics.ClassMetrics

% get images IoU
imageIoU = metrics.ImageMetrics.GlobalAccuracy;

% Find test images with the lowest IoU.
[~, idx] = sort(imageIoU);
idx_bad = idx(1:200);

% Display top five worst results
displayResult(idx_bad(6), 3, imdsTest, pxdsTest, net, cmap, classes);
displayResult(idx_bad(7), 4, imdsTest, pxdsTest, net, cmap, classes);
displayResult(idx_bad(8), 5, imdsTest, pxdsTest, net, cmap, classes);
displayResult(idx_bad(9), 6, imdsTest, pxdsTest, net, cmap, classes);
displayResult(idx_bad(10), 7, imdsTest, pxdsTest, net, cmap, classes);

% Display top five best results
displayResult(1, 8, imdsTest, pxdsTest, net, cmap, classes);
displayResult(2, 9, imdsTest, pxdsTest, net, cmap, classes);
displayResult(9, 10, imdsTest, pxdsTest, net, cmap, classes);
displayResult(40, 11, imdsTest, pxdsTest, net, cmap, classes);
displayResult(23, 12, imdsTest, pxdsTest, net, cmap, classes);

function displayResult(index, figNo, imdsTest, pxdsTest, net, cmap, classes)

imageIndex = index;
testImage = readimage(imdsTest,imageIndex);
trueLabels = readimage(pxdsTest,imageIndex);
% worstPredictedLabels = readimage(pxdsResults,imageIndex);

C = semanticseg(testImage, net);

B = labeloverlay(testImage,C,'Colormap',cmap,'Transparency',0.4);


similarity = bfscore(trueLabels, C); % BF
iou = jaccard(C,trueLabels);
[confus,~,~,~,~,~,~,~,~] = compute_accuracy_F(trueLabels,C,classes);
Accuracy = sum(diag(confus))/sum(confus(:))*100;

figure(figNo);
 set(gcf, 'Color', 'White', 'Unit', 'Normalized', ...
    'Position', [0.1,0.1,0.6,0.6] ) ;

subplot(1,3,1);
imshow(testImage)
subplot(1,3,2);
imshow(B,'XData',[1 1260], 'YData',[1 1020])
title({
    ['\color{red}【 Accuracy 】 \color{black} = ' num2str(Accuracy)]
    ['\color{red}【 BF Score 】 \color{black} TransparentPET = ' num2str(similarity(1)) ' | BluePET = ' num2str(similarity(2)) ' | Black = ' num2str(similarity(3))]
    ['\color{red}【 IoU Score 】 \color{black} TransparentPET = ' num2str(iou(1)) ' | BluePET = ' num2str(iou(2)) ' | Black = ' num2str(iou(3))]
    ['']
    ['']
    [' ']
    });
pixelLabelColorbar(cmap, classes);

actual = uint8(C);
expected = uint8(trueLabels);
subplot(1,3,3); 
imshowpair(actual, expected)

 axes( 'Position', [0, 0.95, 1, 0.05] ) ;
 set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
 text( 0.5, -2, 'Tested Image vs. Prediction vs. Truth', 'FontSize', 18', 'FontWeight', 'Bold', ...
      'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;


end
function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;

end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = SplitData(imds, pxds, classes, labelIDs)

% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);

end

function cmap = ColorMap()
             
             cmap = [
                 255 255 255  % white 
                 0 0 255      % blue
                 0 0 0        % black
                 ];
             
             % Normalize between [0 1].
             cmap = cmap ./ 255;
end

function imds = resizeImages(imds, numTrainingImages)
  reset(imds)

  for c = 1:numTrainingImages
    I = readimage(imds,c);
    [rows, columns, ch] = size(I); 
    if(rows > columns) % vertical image
        I = imrotate(I , 90);
        I = imresize(I,[720 960]);  
    elseif(rows < columns) % horizontal image
        I = imresize(I,[720 960]);
    end   
      
  % Write to disk.
    outputFileName = fullfile('/Users/nujood/Desktop/Bottle-Segmentation/Images_Updated', ['image_' num2str(c) '.png']);
    imwrite(im2uint8(I), outputFileName, 'BitDepth', 8);
   
  end
end
  

function pxds = resizePixelLabels(pxds, numTrainingImages)
  reset(pxds)
  % while hasdata(imds)
  for c = 1:numTrainingImages
    I = readimage(pxds,c);
    [rows, columns, ch] = size(I); 
    if(rows > columns) % vertical image
        I = imrotate(I , 90);
        I = imresize(I,[720 960]);  
    elseif(rows < columns) % horizontal image
        I = imresize(I,[720 960]);
    end  
  % Write to disk.
    outputFileName = fullfile('/Users/nujood/Desktop/Bottle-Segmentation/masks_Updated', ['mask_' num2str(c) '.png']);
    imwrite(im2uint8(I), outputFileName, 'BitDepth', 8);
   
  end
end

function labelIDs = PixelLabelIDs()

labelIDs = {
    % Transparent PET
    [
    255 255 255;
    253 253 253;
    251 251 245;
    251 255 255;
    231 231 241;
    242 241 247
    ]
    
    % Blue PET
    [
    0 0 254;
    0 0 255; 
    85 178 245; 
    74 104 158; 
    0 255 1; 
    0 0 246; 
    0 41 245; 
    0 1 252; 
    16 8 153; 
    18 117 236; 
    17 13 152; 
    23 0 254; 
    0 14 255; 
    0 51 140; 
    0 1 254; 
    2 0 254;
    10 0 255;
    ]
    
    % Black
    [
    0 0 0; 
    1 1 1; 
    1 0 2;
    3 4 0; 
    13 13 3; 
    4 5 0; 
    5 3 1]
    };
end