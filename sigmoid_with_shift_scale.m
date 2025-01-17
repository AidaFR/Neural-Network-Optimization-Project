%Functia sigmoid= 1/(1+exp(-2(z-1))
%Outputul g este o valoare intre (0,1)
% b poate fi utilizat pentru a schimba pragul de activare al unei rețele neurale, 
% iar a poate fi folosit pentru a regla rapiditatea cu care funcția sigmoid se apropie de limitele sale.
function g = sigmoid_with_shift_scale(z)
a = 2;
b = 1;
g = 1 ./ (1 + exp(-a*(z - b)));
end