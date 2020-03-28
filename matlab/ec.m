close all;
layers = get_lenet();


root_dir =('../images/');
real_imgs = dir(strcat(root_dir,'*.*g'));

load lenet.mat
confusion = zeros(10,10);
corrects=0;
threshold = 0.9;
counter =1;

gt= [1,2,3,4,5,6,7,8,9,0,...
    1,-1,2,3,4,5,6,7,8,9,0,...
    6,0,6,6,6,-1,-1,-1,...
    7,0,3,1,1,7,6,2,6,1,...
    3,6,4,9,1,2,4,0,0,5,4,4,...
    7,3,1,2,0,5,5,1,7,7,4,9,...
    1,7,4,2,9,1,3,5,4,2,0,1,9,4,4,1,1];

for i=1:numel(real_imgs)
    image_name = strcat(real_imgs(i).folder,'\',real_imgs(i).name);
    
    test_img = imgaussfilt(imresize(imread(image_name),8));
    test_img = im2double(rgb2gray(test_img)); 
    T = adaptthresh(test_img,0.3,'ForegroundPolarity','dark');
    test_img = double(imbinarize(test_img,T));
     test_img(test_img==1)=10;
     test_img(test_img==0)=1;
     test_img(test_img==10)=0;
    %test_img=double(test_img);
    
    
     %idx =test_img<threshold;
     %test_img = test_img.*idx;
     %test_img(test_img~=0)=1;
     %figure;
     %imshow(test_img,[]);
    CC = bwconncomp(test_img);
    bb = regionprops(CC,'BoundingBox');
    
    num_bb = size(bb,1);
    
    
    for j=1:num_bb
        
        if(gt(counter)==-1)
            counter=counter+1;
            continue;
        end
         
        bb_coord =  bb(j).BoundingBox;
        
        x=bb_coord(1);
        y=bb_coord(2);
        w=bb_coord(3);
        h=bb_coord(4);
        
        x=ceil(x);
        y=ceil(y);
        
        patch = test_img(y:y+h-1,x:x+w-1)';
        %figure;
        %imshow(patch);
        patch  =padarray(patch,[56,56]);
        patch = imresize(patch,[28,28]);
%         figure;
%         imshow(patch');
         patch = patch(:);
         patch = repmat(patch,[1,100]);
         [output, P] = convnet_forward(params, layers,patch);
         [~,pred] = max(P,[],1); 
        % disp(pred(1)-1);
        if(pred(1)==gt(counter)+1)
            corrects=corrects+1;
        end
        confusion(gt(counter)+1,pred(1))=confusion(gt(counter)+1,pred(1))+1;
        counter = counter+1;   
   end
end
fprintf('test accuracy: %f %\n', (corrects/(size(gt,2)-4))*100);