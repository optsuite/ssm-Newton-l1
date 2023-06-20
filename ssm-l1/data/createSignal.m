% see also NESTA package (-> msp_signal)
function [x0,b,Omega,R,Rt] = createSignal(n,m,k,dyna,sig,U,seed)
    sd          = seed + 500 + dyna;
    randn('state',sd); rand('state',sd);

    x0          = zeros(n,1);

    supind      = randperm(n);
    supind      = supind(1:k);

    valx        = dyna/20*(rand(k,1));
    valx        = valx - min(valx); 
    valx        = valx/max(valx)*dyna/20;

    x0(supind)  = 10.^valx.*sign(randn(k,1));

    Omega       = randperm(n);
    Omega       = Omega(1:m);

    R           = @(x) x(Omega,:);

    S.type      = '()'; 
    S.subs{1}   = Omega; 
    S.subs{2}   = ':';

    Rt          = @(x) subsasgn(zeros(n,1),S,x);

    b           = R(U(x0));
    b           = b + sig*randn(m,1);
end