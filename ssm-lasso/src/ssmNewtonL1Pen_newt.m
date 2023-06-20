function [x, out] = ssmNewtonL1Pen_newt(A,At,b,N,mu,opts)

tic;
%%-------------------------------------------------------------------------
if nargin < 5; opts = []; end
if ~isfield(opts,'x0');    opts.x0  = zeros(N,1);   end
if ~isfield(opts,'tol');   opts.tol = 1e-6;     end
if ~isfield(opts,'record');opts.record = 1;     end
if ~isfield(opts,'eps');   opts.eps = 1e-4;     end

if ~isfield(opts,'comp_obj'); opts.comp_obj = 0;   end
if ~isfield(opts,'tau');      opts.tau = 6;   end
if ~isfield(opts,'tau_adapt');opts.tau_adapt = 0;   end
if ~isfield(opts,'tau_m');    opts.tau_m = 1e-3;   end
if ~isfield(opts,'tau_M');    opts.tau_M = 1e+4;   end
if ~isfield(opts,'maxitTau'); opts.maxitTau = 20;   end
if ~isfield(opts,'teta');     opts.teta = 0.85;   end
if ~isfield(opts,'beta');     opts.beta = 1e-1;   end
if ~isfield(opts,'gamma');    opts.gamma = 1e-1;   end
if ~isfield(opts,'CG_maxit'); opts.CG_maxit = 4;   end
if ~isfield(opts,'CG_tol');   opts.CG_tol = 1e-1;   end
if ~isfield(opts,'CG_adapt'); opts.CG_adapt = 1;   end
if ~isfield(opts,'cont');     opts.cont = 1;   end
if ~isfield(opts,'cont_max'); opts.cont_max = 10;   end

if ~isfield(opts,'crit'); opts.crit = 1;   end

x      = opts.x0;
tol    = opts.tol;
record = opts.record;
eps    = opts.eps;
tau    = opts.tau; 

if opts.crit == 2
    fopt = opts.fopt;
end

%     if opts.tol >= 1e-5; opts.CG_maxit = 3;
%     else opts.CG_maxit = 8; end



if opts.tau_adapt
    maxitTau = opts.maxitTau;
    tau_m    = opts.tau_m;
    tau_M    = opts.tau_M;
    teta     = opts.teta;
    Qt       = 1;
end

CG_tol = opts.CG_tol;
CG_maxit = opts.CG_maxit;

%%-------------------------------------------------------------------------
% options for the Levenberg-Marquardt algorithm
if ~isfield(opts,'maxit');    opts.maxit = 100;   end
if ~isfield(opts,'xtol');     opts.xtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts,'muPow');    opts.muPow = 0.5;     end
if ~isfield(opts,'resFac');   opts.resFac = 0.98;   end
if ~isfield(opts,'eta1');     opts.eta1 = 1e-6;   end
if ~isfield(opts,'eta2');     opts.eta2 = 0.9;    end
if ~isfield(opts,'gamma1');   opts.gamma1 = 0.5;  end
if ~isfield(opts,'gamma2');   opts.gamma2 = 1;    end
if ~isfield(opts,'gamma3');   opts.gamma3 = 10;   end
if ~isfield(opts,'lambda');   opts.lambda = 0.1;    end %adjust mu
if ~isfield(opts,'itPrint');  opts.itPrint = 1;   end
if ~isfield(opts,'maxItStag');opts.maxItStag = 5; end
if ~isfield(opts,'restart');  opts.restart = 1; end
if ~isfield(opts,'maxItPCG'); opts.maxItPCG = 20;   end

maxItStag = opts.maxItStag; restart=opts.restart; maxItPCG=opts.maxItPCG;
maxit   = opts.maxit;   itPrint = opts.itPrint; muPow   = opts.muPow;
xtol    = opts.xtol;    ftol    = opts.ftol;    resFac  = opts.resFac;
eta1    = opts.eta1;    eta2    = opts.eta2;    lambda  = opts.lambda;
gamma1  = opts.gamma1;  gamma2  = opts.gamma2;  gamma3  = opts.gamma3;
%-------------------------------------------------------------------------

%------------------------------------------------------------------
if record >= 1
    % set up print format
    if ispc; str1 = '  %12s'; str2 = '  %8s';
    else     str1 = '  %12s'; str2 = '  %8s'; end
    stra_FP = ['switch to the fixed-point method\n','%5s',str1,str2,'\n'];
    str_head_FP = sprintf(stra_FP, 'iter', 'res', 'resr');
    str_num_FP = ['%4d  %+8.7e  %+2.1e\n'];
    str_num_LBFGS = ['%4d  %+8.7e  %+2.1e   %d   %2.1e   %2.1e   %d\n'];
    stra = ['switch to the Newton method\n',...
        '%5s','%6s', str1, str2,str2, str2, str2, str2, str2, str1, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'mu', 'res', 'resR', 'Ftxzd', 'rhs', 'ratio', ...
        'sig','lambda','CG(flag/it/res)');
    str_num = ['%4d  %+2.1e %+8.7e  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %+2.1e ' ...
        '  %+2.1e %1d %2d %2.1e %2s\n'];
end

% continuation
muf     = mu;

if opts.cont
    bsup        = max(abs(b));
    mu0         = min(0.25,2.2*(bsup/muf)^(-1/3))*bsup;
    mu          = max(muf,mu0);
    
    cont_max    = opts.cont_max;
    cont_count  = 1;
end

% tolerances
if opts.CG_adapt
    res_tol = 1e-3;
end
res         = 1e20;
ftol        = 1e-4;

% first computations
Atb         = At(b);
btb         = b'*b;

% compute residual function p = F_tau(x)
[Ftx, res, yx, grad] = comp_res(x);
xp = x;  resp = res; 
if opts.tau_adapt; gradp = grad; end
if opts.cont; resp_mu = res; end

% stage two: Levenberg-Marquardt algorithm
% itStag = 0; iterd = 1; flag = []; relres = [];

nr_CG = 0;
nr_res = 1;
nr_Acall = 3;

itTau = 0;
switch_pow = 1;
for iter = 1:maxit
    itTau = itTau + 1;

    %----------------------------------------------------------------------
    % parameter for the revised Newton system
    if switch_pow
        if res/tau < 1e-9
            muPow = 0.15;   %60dB 0.1 20dB 0.15
        elseif res/tau < 1e-3
            muPow = 0.6;
        elseif res/tau < 1e-2
            muPow = 0.6;
        elseif res/tau < 10/6
            muPow = 0.7;
        else
            muPow = 0.7;
        end
    end
    
    sig = lambda*(resp^muPow);
    %sig = lambda*min(1,resp)^muPow;
    
    ind     = abs(yx) <= (1+eps)*tau*mu;
    eps     = 0.1*eps;
    % reduce the Newton system and solve it via CG
    d           = 0*x;
    d(ind)      = -Ftx(ind)/(1+sig);
    %d(ind)      = -Ftx(ind)/(1/tau+sig);

    AtAd        = At(A(d));
    nr_Acall = nr_Acall + 2;
    r           = - Ftx/tau - AtAd;
    %r           = - Ftx - AtAd;
    Mat         = @(y)build_CGMatrix(ind,y);
        
    [d(~ind),iterd,flag,~,~,CG_AtAd,relres] = ...
        mypcgw(Mat,r(~ind),CG_tol,CG_maxit,[],1);
    
    nr_CG = nr_CG + iterd;
    nr_Acall = nr_Acall + iterd*2;
    %----------------------------------------------------------------------
    
    xz = x + d;
    gradxz = grad + AtAd + CG_AtAd;
    [Ftxz, resxz, yxz] = comp_resR(xz,gradxz);
    %[Ftxz, resxz, yxz, objz] = comp_resR(xz,gradxz);
    nrmd = norm(d(:));
    Ftxzd = -idot(Ftxz,d);
    
    if res < 1e-2; rhs = resxz*nrmd^2; else rhs = nrmd^2; end
    %rhs = nrmd^2;
    
    ratio = Ftxzd/rhs;  %ratio = 1;
    resRatiop = resxz/resp;
    
    success = ratio >= eta1; ddflag = 'f';
    % Newton step
    x = xz; Ftx = Ftxz; res = resxz; ddflag = 'n';
    grad = gradxz; yx = yxz;  %obj = objz;
    
    
    % stopping criteria, res/tau is comparable to SNF.m
    switch opts.crit
        case 1
            cstop = (mu <= (1+1e-10)*muf) && (res/tau <= tol);
        case 2
            if mu <= (1+1e-10)*muf 
                fiter = 0.5*((grad-Atb)'*x + btb) + mu*norm(x,1);
                cstop = ((fiter - fopt)/max(1,abs(fopt)) <= tol);
            else
                cstop = 0;
            end 
    end
    
    
    if record>=1 && (cstop || ...
            iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit && ~cstop;
            fprintf('\n%s', str_head);
        end
        fprintf(str_num, iter, mu, res/tau, resRatiop, ...
            Ftxzd, rhs, ratio, sig,lambda, flag, iterd, relres, ddflag);
    end
    
    
    if cstop
        if opts.crit == 2
            out.f = fiter;
        end
        out.msg = 'optimal';
        break;
    end
    
    if ratio >= eta2
        lambda = max(gamma1*lambda, 1e-16);
    elseif ratio >= eta1
        lambda = gamma2*lambda;
    else
        if ratio >0
            lambda = gamma3*lambda;
        else
            lambda = 2*gamma3*lambda;
        end
    end
    
    if success
        % evaluate the quality of the last iterate
        if opts.cont
            goodIter = (cont_count >= cont_max) ||  ...
             ((res <= 0.5*resp) || (res <= max(0.1*resp_mu, ftol)));
        else
            goodIter = (res <= 0.5*resp)|| (res <= max(0.1*resp_mu, ftol));
        end
        
        % update tau
        mu_tau_change = 0;
        if opts.tau_adapt
            if iter >= 2
                if goodIter && mu > muf
                    tau     = opts.tau;
                elseif itTau >= maxitTau
                    %  estimation of Lipschitz constant for smooth term.
                    ng      = norm(x - xp)/norm(grad - gradp);
                    Qt      = teta*Qt + 1;
                    % convex combination with previous tau
                    tau     = ((Qt-1)*tau + max(min(ng,tau_M),tau_m))/Qt;
                    mu_tau_change = 1;
                    itTau = 0;
                end
            end
        end
    
        % continuation: update mu
        if opts.cont
            if goodIter
                mup         = mu;
                muredf      = max((1-(0.65^log10(mu0/mu))*0.535),0.15);
                mu          = max(muf,muredf*mu);
                
                if abs(mu-mup) > 1e-12; mu_tau_change = 1; end
                
                if mu > muf
                    cont_count      = 1;
                else
                    cont_count      = cont_count + 1;
                end
            else
                cont_count = cont_count + 1;
            end
        end
        
        if mu_tau_change; [Ftx, res, yx]  = comp_resR(x,grad); end
        if opts.cont  && mu_tau_change; resp_mu = res; end

     %   update CG parameters if final mu is reached
        if opts.CG_adapt && (mu <= (1+1e-10)*muf)
            if res < res_tol;
                CG_tol     = max(0.08*CG_tol,1e-7);
                res_tol    = 0.1*res_tol;
            end
            
            if res/tau < 1e-4;
                CG_maxit   = min(max(CG_maxit + 7,3),70);
            elseif res/tau < 1e-3;
                CG_maxit   = 7;
            elseif res/tau < 1e-2;
                CG_maxit   = 6;
            else
                CG_maxit   = 5;
            end
        end

%         if opts.CG_adapt && (mu <= (1+1e-10)*muf)
%             if res < res_tol;
%                 CG_maxit   = min(max(CG_maxit + 7,5),50);
%                 CG_tol     = max(1e-1*CG_tol,1e-6);
%                 res_tol    = 1e-1*res_tol;
%             end
%         end
        resp = res; xp = x; 
        if opts.tau_adapt; gradp = grad; end
    end
    
end %end outer loop
out.time = toc;

out.iter = iter;
out.res  = res;

out.nr_res = nr_res;
out.nr_CG = nr_CG;
out.Acalls = nr_Acall;% + nr_res*2 + nr_CG*2;

    function a = idot(x,y)
        a = real(x(:)'*y(:));
    end

    % computes a reduced generalized derivative for the CG method
    function [z,BtBr] = build_CGMatrix(ind,y)
        z           = zeros(N,1);
        z(~ind)     = y;
        BtBr        = At(A(z));
        z           = BtBr(~ind) + (sig/tau)*y;
        %z           = BtBr(~ind) + sig*y;
    end

    function ss = shringkage(xx,mumu)
        ss = sign(xx).*max(abs(xx)-mumu,0);
    end

%     function [Ftx, res, yx, grad, obj] = comp_res(xx)
%         grad   = At(A(xx)) - Atb;
%         yx     = xx - tau*grad;
%         Ftx    = xx - shringkage(yx,mu*tau);
%         %res    = norm(Ftx); 
%         res    = norm(Ftx)/tau;        
%         obj    = mu*norm(x,1) + 0.5*res^2;
%     end
% 
%     function [Ftx, res, yx, obj] = comp_resR(xx,grad)
%         yx     = xx - tau*grad;
%         Ftx    = xx - shringkage(yx,mu*tau);
%         %res    = norm(Ftx);
%         res    = norm(Ftx)/tau;
%         obj    = mu*norm(x,1) + 0.5*res^2;
%     end

    function [Ftx, res, yx, grad] = comp_res(xx)
        grad   = At(A(xx)) - Atb;
        yx     = xx - tau*grad;
        Ftx    = xx - shringkage(yx,mu*tau);
        res    = norm(Ftx); 
    end

    function [Ftx, res, yx] = comp_resR(xx,grad)
        yx     = xx - tau*grad;
        Ftx    = xx - shringkage(yx,mu*tau);
        res    = norm(Ftx);
    end

end



