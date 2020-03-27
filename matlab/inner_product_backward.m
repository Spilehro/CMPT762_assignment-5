function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));
input_od = zeros(size(input.data));

param_grad.b = sum(output.diff,2)';
param_grad.w = (output.diff*input.data')';

batch_size = input.batch_size;
for i=1:batch_size
    input_od(:,i) = param.w*output.diff(:,i);
end
end
