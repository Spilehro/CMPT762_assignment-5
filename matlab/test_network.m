%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat
confusion = zeros(100,1);
confusion2 = zeros(10,10);
confusion3 = zeros(10,10);

%% Testing the network
% Modify the code to get the confusion matrix
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [~,pred] = max(P,[],1);
    gt = ytest(i:i+99);
    idx = sub2ind([10,10],gt,pred);
    confusion(idx)=confusion(idx)+1;
    confusion3(gt,pred)=confusion3(gt,pred)+1;
    for j=1:size(pred,2)
       confusion2(gt(j),pred(j))=confusion2(gt(j),pred(j))+1;
    end
end
confusion =reshape(confusion,[10,10]);