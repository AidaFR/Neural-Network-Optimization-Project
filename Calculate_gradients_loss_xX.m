function [grad_loss_x, grad_loss_X] = Calculate_gradients_loss_xX(e, y, X, x, A)
disp(size(e))
N = length(y);  
    sum=0;
    for i = 1:N
        sum=sum+(e(i)/ y(i) - ((1 - e(i)) / (1 - y(i))));
    end
    sum=-sum/N;
        a_i = A(i, :);
        %sigmoid_val = X'*A';
        grad_loss_x = sum*sigmoid_with_shift_scale(A*X);
        grad_loss_X = sum*sigmoid_deriv( A*X)*x;
    
end