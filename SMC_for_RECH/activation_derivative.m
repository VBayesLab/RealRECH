function output = activation_derivative(z,text)

    switch text
        case 'Linear'
            output = 1;
        case 'Sigmoid'
            output = 1.0 ./ (1.0 + exp(-z));
            output = output.*(1-output);
        case 'Tanh'
            output = exp(-2*z); 
            output = (1-output)./(1+output); 
            output = 1-output.^2;
        case 'ReLU'
            output = ones(size(z)); 
            output(z<=0) = 0;
        case 'Softplus'
            output = 1./(1+exp(-z));            
    end
end