function L = Loss(e,y)
L=0;
N=length(e);
    for i=1:N
    L=L+(e(i)*log(y(i))+(1-e(i))*log(1-y(i)));
    end
L=L*(-1/N);
end

