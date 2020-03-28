layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
%imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);

conv_h =output{2}.height;
conv_w=output{2}.width;
conv_channel=output{2}.channel;
output_conv = reshape(output{2}.data,[conv_h conv_w conv_channel]);

relu_h =output{3}.height;
relu_w=output{3}.width;
relu_channel=output{3}.channel;
output_relu = reshape(output{3}.data,[relu_h relu_w relu_channel]);

% Fill in your code here to plot the features.
for i=1:conv_channel
    figure(1);
    subplot(4,5,i); imshow(output_conv(:,:,i)');
    figure(2);
    subplot(4,5,i); imshow(output_relu(:,:,i)');
end
