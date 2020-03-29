        % Network defintion
layers = get_lenet();
close all;
% Loading data
%fullset = false;
%[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
root_dir =('../RealWord/');
real_imgs = dir(strcat(root_dir,'*.*g'));
gt = [1,2,3,4,5,6,7,8,9,10];
%gt=[2,2,2,2,2,2,2,2];

load lenet.mat
confusion = zeros(10,10);
corrects=0;
figure;
for i=1:numel(real_imgs)
    image_name = strcat(real_imgs(i).folder,'\',real_imgs(i).name);
    test_img = imread(image_name);
    test_img= im2double(rgb2gray(test_img));
    subplot(2,5,i);imshow(test_img);
    test_img = imresize(imcomplement(test_img),[28,28])'; 
    test_img = test_img(:);
    test_img = repmat(test_img,[1,100]);
    [output, P] = convnet_forward(params, layers,test_img);
    [~,pred] = max(P,[],1);
    pred_num = pred(1);
    gt_num = gt(i);
    if(pred_num==gt_num)
        corrects=corrects+1;
    end
    confusion (gt_num,pred_num)=confusion(gt_num,pred_num)+1;
   

end
fprintf('test accuracy: %d %\n', (corrects/10)*100);
