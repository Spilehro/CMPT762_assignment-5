function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;
    
    data = reshape(input.data,[h_in,w_in,c,batch_size]);

    % Replace the following line with your implementation.
    k_size=floor(k/2); 
    output.data = zeros([h_out, w_out, c, batch_size]);
    max_val_store = zeros(h_out*w_out,1);
    counter=1;
    if(k~=stride)
        for b=1:batch_size
            batch=data(:,:,:,b);
            for ci=1:c
                chan = batch(:,:,ci);
                for hi=1:stride:h_in-k_size
                    for wi=1:stride:w_in-k_size
                        max_val_store(counter) = max(chan(hi:hi+k_size,wi:wi+k_size),[],'all');
                        counter=counter+1;
                    end
                end
                counter=1;
                output.data(:,:,ci,b)=reshape(max_val_store,[h_out,w_out])';
            end
        end
    else
         for i=1:batch_size
            output.data(:,:,:,i)= sepblockfun(padarray(data(:,:,:,i),[pad pad]),[k, k],'max');
            %output.data(:,:,:,i) = pooled(1:stride:end, 1:stride:end,:);  
         end
    end
     output.data = reshape(output.data,[h_out*w_out*c,batch_size]);
    
end

