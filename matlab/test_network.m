%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat
confusion = zeros(10,10);
%confusion2 = zeros(10,10);

%% Testing the network
% Modify the code to get the confusion matrix
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [~,pred] = max(P,[],1);
    gt = ytest(i:i+99);
    for j = 1:size(pred,2)
        confusion(gt(j),pred(j)) = confusion(gt(j),pred(j))+1;
    end
   
end
%confusion =reshape(confusion,[10,10]);