%Download the CamVid Dataset
imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';

outputFolder = fullfile('G:\Sankar\Jijnasa\fcn_matlab', 'CamVid');

if ~exist(outputFolder, 'dir')
    disp('Downloading 557 MB CamVid dataset...');

    unzip(imageURL, fullfile(outputFolder,'images'));
    unzip(labelURL, fullfile(outputFolder,'labels'));
end

%Use imageDatastore to load CamVid images
imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);

%Display one of the image
I = readimage(imds, 1);
I = histeq(I);
figure
imshow(I)

%Load CamVid Pixel-Labeled Images
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

%Return the grouped label IDs 
labelIDs = camvidPixelLabelIDs();

%Use the classes and label IDs to create the pixelLabelDatastore
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

%Read and display one of the pixel-labeled images by overlaying it on top of an image.
C = readimage(pxds, 1);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
figure
imshow(B)
pixelLabelColorbar(cmap,classes);

 %The distribution of class labels in the CamVid dataset
 tbl = countEachLabel(pxds)
 
 %Visualize the pixel counts by class
 frequency = tbl.PixelCount/sum(tbl.PixelCount);
 figure
 bar(1:numel(classes),frequency)
 xticks(1:numel(classes))
 xticklabels(tbl.Name)
 xtickangle(45)
 ylabel('Frequency')
 
 %Resize CamVid Data to reduce the training time
imageFolder = fullfile(outputFolder,'imagesReszed',filesep);
imds = resizeCamVidImages(imds,imageFolder);
labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizeCamVidPixelLabels(pxds,labelFolder);

%partition training and testing data
[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds);

%display the number of training images and number of testing images
numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

%create the network
imageSize = [360 480 3];
numClasses = numel(classes);
lgraph = fcnLayers(imageSize,numClasses) %fcn-8s

%Balance Classes Using Class Weighting so as to get proper training results
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

%Specify the class weights using a pixelClassificationLayer.
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

%Update the SegNet network with the new pixelClassificationLayer by
%removing the current pixelClassificationLayer and adding the new layer and
%the new layer to the network
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');


%stochastic gradient decent with momentum (SGDM) is used for training and
%its hyperparameters are specified
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 5, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 2);

%Data augmentation is used during training to provide more examples to the network because it helps improve the accuracy of the network
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);

%pixelLabelImageSource reads batches of training data, applies data augmentation, and sends the augmented data to the training algorithm
datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

%Startl training using trainNetwork if the doTraining flag is true
%otherwise load pretrained model
doTraining = true;
if doTraining
    [net, info] = trainNetwork(datasource,lgraph,options);
end

%run the trained model on a test image
I = read(imdsTest);
C = semanticseg(I, net);

%display the result
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);

 