function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')

weights = reshape(param.w,[k,k,c,num]);
data = reshape(input.data,[h_in,w_in,c,batch_size]);
output.data = zeros([h_out, w_out, num, batch_size]);
%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 
for i=1:batch_size
   A = data(:,:,:,i);
   A = padarray(A,[pad,pad],0);
   for j=1:num
       output.data(:,:,j,i) = convn(A,flip(flip(flip(weights(:,:,:,j),1),2),3),'valid')+param.b(j);
   end    
end
output.height = h_out;
output.width = w_out;
output.channel = num;
output.data = reshape(output.data,[h_out*w_out*num,batch_size]);
output.batch_size = batch_size;


end

