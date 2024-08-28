% Load the MNIST dataset
[XTrain,YTrain] = digitTrain4DArrayData;
[XTest,YTest] = digitTest4DArrayData;

% Define the network architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(32,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(32,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(64,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(64,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(128,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(128,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(512)
    reluLayer
    
    dropoutLayer(0.5)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Specify the training options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',15, ...
    'MiniBatchSize',128, ...
    'Plots','training-progress');

% Training
net = trainNetwork(XTrain, YTrain, layers, options);

% Testing
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest)/numel(YTest)

% Compute and Display Confusion Matrix
confMat = confusionmat(YTest, YPred);
disp(confMat);
