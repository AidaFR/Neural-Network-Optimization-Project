%Functia sigmoid derivata = 2*exp(-2(z-1))/(1+exp(-2(z-1))^2
function gd = sigmoid_deriv(z)
a = 2;
b = 1;
gd = 2 *exp(-a*(z-b))./(1 + exp(-a*(z-b))).^2;
end