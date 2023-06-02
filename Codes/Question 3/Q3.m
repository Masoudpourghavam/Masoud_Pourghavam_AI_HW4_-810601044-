% Masoud Pourghavam
% Student Number: 810601044
% Course: Artificial Intelligence
% University of Tehran
% Homework 4

%% Initialization
clc;
clear all;
close all;

%% Read the CSV file
data = readtable('dataset_2.csv');

%% Replace 'No Failure' with 0 and other labels with 1 in column 9
data.FailureType = double(~strcmp(data.FailureType, 'No Failure'));

%% Save the updated dataset
outputFile = 'updated_dataset.csv';
writetable(data, outputFile);

%% Read the CSV file
data = readmatrix('updated_dataset.csv');

% Display the modified dataset
disp(data);

%% Select the input data (X) from the fourth to eighth columns
x = data(:, 4:8);

%% Select the output data (Y) from the ninth column
y = data(:, 9);

%% Standardize the data
x = zscore(x);

%% Visualize the data
histogram(y,10);
title ('Distribution of classes')
xlabel ('Class');
ylabel ('Frequency');

%% Train an ANN
xt = x';
yt = y';
hiddenLayerSize = 10;  % Number of neurons in the hidden layer
net = fitnet(hiddenLayerSize, 'traingdm');
net.layers{1}.transferFcn = 'poslin';  % Set ReLU as the activation function for the hidden layer
net.divideParam.trainRatio = 80/100;  % 80% for training
net.divideParam.valRatio = 0/100; % 0% for training
net.divideParam.testRatio = 20/100; % 20% for training
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 20;
net.trainParam.lr = 0.1;
net.trainFcn = 'trainlm';  % Levenberg-Marquardt optimizer
net.performFcn = 'crossentropy';  % Appropriate loss function for classification
[net,tr] = train(net, xt , yt);

%% Performance of the ANN
fprintf('################ Neurons = 10 ################');
yTrain10 = net(xt(:,tr.trainInd));
yTrainTrue10 = yt(tr.trainInd);
trainRMSE10 = sqrt(mean((yTrain10 - yTrainTrue10).^2));
trainRMSE10

yTest10 = net(xt(:,tr.testInd));
yTestTrue10 = yt(tr.testInd);
testRMSE10 = sqrt(mean((yTest10 - yTestTrue10).^2));
testRMSE10

%{

%% Optimize the number of neurons in the hidden layer

for i = 1:60

    % Defining the artichectute of the ANN
    hiddenLayerSize = i;  % Number of neurons in the hidden layer
    net = fitnet(hiddenLayerSize, 'traingdm');
    net.layers{1}.transferFcn = 'poslin';  % Set ReLU as the activation function for the hidden layer
    net.divideParam.trainRatio = 80/100;  % 80% for training
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 20/100;
    net.trainParam.epochs = 1000;
    net.trainParam.max_fail = 20;
    net.trainParam.lr = 0.1;
    net.trainFcn = 'trainlm';  % Levenberg-Marquardt optimizer
    net.performFcn = 'crossentropy';  % Appropriate loss function for classification

    % Training the ANN
    [net,tr] = train(net, xt , yt);

    % Determine the error of the ANN
    yTrain = net(xt(:,tr.trainInd));
    yTrainTrue = yt(tr.trainInd);
    trainRMSE(i) = sqrt(mean((yTrain - yTrainTrue).^2)); % RMSE of training set

    yTest = net(xt(:,tr.testInd));
    yTestTrue = yt(tr.testInd);
    testRMSE(i) = sqrt(mean((yTest - yTestTrue).^2)); % RMSE of testing set

end


%% Select the optimal number of neurons in hidden layer
plot (1:60, trainRMSE); hold on;
plot (1:60, testRMSE); hold off; 
legend ('trainRMSE', 'testRMSE');
xlabel('Number of hidden layer neurons');
ylabel('RMSE');

%}

%% Train an ANN with 1 neuron
xt = x';
yt = y';
hiddenLayerSize = 1;  % Number of neurons in the hidden layer
net = fitnet(hiddenLayerSize, 'traingdm');
net.layers{1}.transferFcn = 'poslin';  % Set ReLU as the activation function for the hidden layer
net.divideParam.trainRatio = 80/100;  % 80% for training
net.divideParam.valRatio = 0/100; % 0% for training
net.divideParam.testRatio = 20/100; % 20% for training
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 20;
net.trainParam.lr = 0.1;
net.trainFcn = 'trainlm';  % Levenberg-Marquardt optimizer
net.performFcn = 'crossentropy';  % Appropriate loss function for classification
[net,tr] = train(net, xt , yt);

% Performance of the ANN with 1 neuron
fprintf('################ Neurons = 1 ################');
yTrain1 = net(xt(:,tr.trainInd));
yTrainTrue1 = yt(tr.trainInd);
trainRMSE1 = sqrt(mean((yTrain1 - yTrainTrue1).^2));
trainRMSE1


yTest1 = net(xt(:,tr.testInd));
yTestTrue1 = yt(tr.testInd);
testRMSE1 = sqrt(mean((yTest1 - yTestTrue1).^2));
testRMSE1

%% Train an ANN with 30 neurons
xt = x';
yt = y';
hiddenLayerSize = 30;  % Number of neurons in the hidden layer
net = fitnet(hiddenLayerSize, 'traingdm');
net.layers{1}.transferFcn = 'poslin';  % Set ReLU as the activation function for the hidden layer
net.divideParam.trainRatio = 80/100;  % 80% for training
net.divideParam.valRatio = 0/100; % 0% for training
net.divideParam.testRatio = 20/100; % 20% for training
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 20;
net.trainParam.lr = 0.1;
net.trainFcn = 'trainlm';  % Levenberg-Marquardt optimizer
net.performFcn = 'crossentropy';  % Appropriate loss function for classification
[net,tr] = train(net, xt , yt);

% Performance of the ANN with 30 neurons
fprintf('################ Neurons = 30 ################');
yTrain30 = net(xt(:,tr.trainInd));
yTrainTrue30 = yt(tr.trainInd);
trainRMSE30 = sqrt(mean((yTrain30 - yTrainTrue30).^2));
trainRMSE30


yTest30 = net(xt(:,tr.testInd));
yTestTrue30 = yt(tr.testInd);
testRMSE30 = sqrt(mean((yTest30 - yTestTrue30).^2));
testRMSE30

%% Train an ANN with 500 neurons
xt = x';
yt = y';
hiddenLayerSize = 500;  % Number of neurons in the hidden layer
net = fitnet(hiddenLayerSize, 'traingdm');
net.layers{1}.transferFcn = 'poslin';  % Set ReLU as the activation function for the hidden layer
net.divideParam.trainRatio = 80/100;  % 80% for training
net.divideParam.valRatio = 0/100; % 0% for training
net.divideParam.testRatio = 20/100; % 20% for training
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 20;
net.trainParam.lr = 0.1;
net.trainFcn = 'trainlm';  % Levenberg-Marquardt optimizer
net.performFcn = 'crossentropy';  % Appropriate loss function for classification
[net,tr] = train(net, xt , yt);

% Performance of the ANN with 500 neurons
fprintf('################ Neurons = 500 ################');
yTrain500 = net(xt(:,tr.trainInd));
yTrainTrue500 = yt(tr.trainInd);
trainRMSE500 = sqrt(mean((yTrain500 - yTrainTrue500).^2));
trainRMSE500


yTest500 = net(xt(:,tr.testInd));
yTestTrue500 = yt(tr.testInd);
testRMSE500 = sqrt(mean((yTest500 - yTestTrue500).^2));
testRMSE500

%% Train an ANN with 20 neurons
xt = x';
yt = y';
hiddenLayerSize = 20;  % Number of neurons in the hidden layer
net = fitnet(hiddenLayerSize, 'traingdm');
net.layers{1}.transferFcn = 'poslin';  % Set ReLU as the activation function for the hidden layer
net.divideParam.trainRatio = 80/100;  % 80% for training
net.divideParam.valRatio = 0/100; % 0% for training
net.divideParam.testRatio = 20/100; % 20% for training
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 20;
net.trainParam.lr = 0.1;
net.trainFcn = 'trainlm';  % Levenberg-Marquardt optimizer
net.performFcn = 'crossentropy';  % Appropriate loss function for classification
[net,tr] = train(net, xt , yt);

% Performance of the ANN with 20 neurons
fprintf('################ Neurons = 20 ################');
yTrain20 = net(xt(:,tr.trainInd));
yTrainTrue20 = yt(tr.trainInd);
trainRMSE20 = sqrt(mean((yTrain20 - yTrainTrue20).^2));
trainRMSE20


yTest20 = net(xt(:,tr.testInd));
yTestTrue20 = yt(tr.testInd);
testRMSE20 = sqrt(mean((yTest20 - yTestTrue20).^2));
testRMSE20

