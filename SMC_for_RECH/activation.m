function output = activation(z,text)
% if strcmp(type,'tanh')    
%     f = exp(-2*z); f = (1-f)./(1+f);
% end
% if strcmp(type,'ReLU')
%     f = max(0,z);
% end

% Calculate activation output of hidden units
% z: pre-activation of current hidden unit -> can be a scalar or array
% vector (all units of a single hidden layer)
% text: specified activation function
% text = {Sigmoid, Tanh, ReLU, LeakyReLU, Maxout}

    switch text
        case 'Linear'
            output = z;
        case 'Sigmoid'
            output = 1.0 ./ (1.0 + exp(-z));
        case 'Tanh'
            output = tanh(z);
        case 'ReLU'
            output = max(0,z);
        case 'Softplus'
            %output = log(1+exp(z));
            output = zeros(size(z));
            aux1 = log(1+exp(z));
            ind1 = aux1 == Inf;
            part1 = z(ind1);    % pick up the part of z that is too large 
            output(ind1) = part1+log(1+exp(-part1));

            part2 = z(~ind1);    
            output(~ind1) = log(1+exp(part2));
%        case 'LeakyReLU'
%            output = max(0,z)+ alpha*min(0,z);
%        case 'Maxout'
            
    end
end
