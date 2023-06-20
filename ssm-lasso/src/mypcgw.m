function [x,it,info,r,p,Ax,rnm] = mypcgw(A,b,rtol,maxit,precf,Axreturned)
%[x,it,info,r,p,Ax] = mypcg(A,b,rtol,maxit,precf,Axreturned)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MYPCG, v1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Conjugate gradient method that solves A(x) = b with preconditioner
% 'precf'. If Axreturned is 1, it is assumed that by calling A(x), also a 
% vector 'Mx' is computed and returned.
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%
% Copyright (c) 2013. Michael Ulbrich
% Technische Universitaet Muenchen, milzarek@ma.tum.de, mulbrich@ma.tum.de
%
%   For details on the license see the SNF.m file
%
% last modification: Juli, 2013
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if nargin >= 5
    precavail   = ~isempty(precf);
else
    precavail   = false;
end

if nargin < 6
    Axreturned  = false;
end

x       = 0*b;
Ax      = 0;
r       = b;
bnm     = norm(b);
rnm     = bnm; 

if rnm > 1e16 
    return
end

info = 1;

if bnm == 0
    it  = 0;
    p   = r;
    return
end

if precavail
    z   = feval(precf,r);
else
    z   = r;
end

p       = z;
rho     = r'*z;

for it = 1:maxit

    if Axreturned
        [q,Mp]  = feval(A,p);
    else
        q       = feval(A,p);
    end
    
    ptq         = p'*q;
 
    if ptq <= 0
        info    = 2;
        return
    end
 
    al          = rho/ptq;
    x           = x + al*p;
    r           = r - al*q;
    
    if Axreturned
        Ax      = Ax + al*Mp;
    end

    rnm         = norm(r);
 
    if (rnm <= rtol*bnm)
        %fprintf(1,'normal termination\n');
        return
    end 
 
    if precavail
        z       = feval(precf,r);
    else
        z       = r;
    end
    
    rtz         = r'*z;
    beta        = rtz/rho;
    p           = z+beta*p;
    rho         = rtz;
end

info = 3;
