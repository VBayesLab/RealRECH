function z = activation_inv(f,text)
    switch text
        case 'Linear'
            z = f;
        case 'Sigmoid'
            z = log(f./(1-f));
        case 'Tanh'
            z = (1+f)./(1-f);
            z = 1/2*log(z);           
        case 'Softplus'
            z = log(exp(f)-1);                
    end
end
