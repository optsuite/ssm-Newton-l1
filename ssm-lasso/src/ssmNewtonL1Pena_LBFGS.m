function [x, out] = ssmNewtonL1Pena_LBFGS(A,At,b,N,mu,opts)

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
if ~isfield(opts,'CG_maxit'); opts.CG_maxit = 2;   end
if ~isfield(opts,'CG_tol');   opts.CG_tol = 1e-1;   end
if ~isfield(opts,'CG_adapt'); opts.CG_adapt = 1;   end
if ~isfield(opts,'cont');     opts.cont = 1;   end
if ~isfield(opts,'cont_max'); opts.cont_max = 10;   end
if ~isfield(opts,'nnewtcomp'); opts.nnewtcomp = 10;   end

if ~isfield(opts,'crit'); opts.crit = 1;   end

x      = opts.x0;
tol    = opts.tol;
record = opts.record;
eps    = opts.eps;
tau    = opts.tau;
nnewtcomp = opts.nnewtcomp;
resnp = zeros(nnewtcomp,1);
rnpinx = 0;

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

if ~isfield(opts,'mm');       opts.mm   = 5;      end
mm = opts.mm;

n = length(x);

% set up storage for L-BFGS
SK = zeros(n,mm);	% S stores the last ml changes in x
YK = zeros(n,mm);	% Y stores the last ml changes in gradient of f.
STY = zeros(mm);
STS = zeros(mm);
YTY = zeros(mm);
istore = 0; pos = 0;  ppos = 0;  perm = []; flag = 1; relres = 0;
hmu = 1;

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
    str_num = ['%4d  %+2.1e %+8.7e  %+7.6e  %+2.1e  %+2.1e  %+2.1e  %+2.1e ' ...
        '  %+2.1e %1d %2.1e %2s\n'];
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

max_mu = max(abs(Atb(:)));
firstmuFactor = 0.8*max_mu / mu;
if (mu >= max_mu)
    x = zeros(N,1);
    out.f = 0.5*btb;
    return
end

% compute residual function p = F_tau(x)
[Ftx, res, yx, grad] = comp_res(x);
xp = x;  resp = res; Ftxp = Ftx;
resnp(mod(rnpinx,nnewtcomp)+1) = res;
rnpinx = rnpinx+1;
if opts.tau_adapt; gradp = grad; end
if opts.cont; resp_mu = res; end
f = 0.5*((grad-Atb)'*x + btb) + mu*norm(x,1);

% continuation
final_stopCriterion =opts.crit;
muf     = mu;
if opts.cont
    bsup        = max(abs(b));
    %mu0         = min(0.25,2.2*(bsup/muf)^(-1/3))*bsup;
    %mu          = max(muf,mu0);
    mu = realmax;
    adjust_mu(grad);
end


% stage two: Levenberg-Marquardt algorithm
% itStag = 0; iterd = 1; flag = []; relres = [];

nr_CG = 0;
nr_res = 1;
nr_Acall = 3;

itTau = 0;
switch_pow = 1;

for iter = 1:maxit
    itTau = itTau + 1;
    fp = f;
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
            muPow = 0.8;
        end
    end
    muPow = 0.7;
    
    sig = lambda*(resp^muPow);
    %sig = lambda*min(1,resp)^muPow;
    
%     ind     = abs(yx) <= (1+eps)*tau*mu;
%     eps     = 0.1*eps;
%     % reduce the Newton system and solve it via CG
%     d           = 0*x;
%     d(ind)      = -Ftx(ind)/(1+sig);
%     %d(ind)      = -Ftx(ind)/(1/tau+sig);
%     
%     AtAd        = At(A(d));
%     nr_Acall = nr_Acall + 2;
%     r           = - Ftx/tau - AtAd;
%     %r           = - Ftx - AtAd;
%     Mat         = @(y)build_CGMatrix(ind,y);
    
%     [d(~ind),iterd,flag,~,~,CG_AtAd,relres] = ...
%         mypcgw(Mat,r(~ind),CG_tol,CG_maxit,[],1);
% 
%     nr_CG = nr_CG + iterd;
%     nr_Acall = nr_Acall + iterd*2;
    %----------------------------------------------------------------------
    if iter == 1
      %  d = Hlrinv_no(-Ftx);
%         ind     = abs(yx) <= (1+eps)*tau*mu;
%         eps     = 0.1*eps;
%         % reduce the Newton system and solve it via CG
%         d           = 0*x;
%         d(ind)      = -Ftx(ind)/(1+sig);
%         %d(ind)      = -Ftx(ind)/(1/tau+sig);
% 
%         AtAd        = At(A(d));
%         nr_Acall = nr_Acall + 2;
%         r           = - Ftx/tau - AtAd;
%         %r           = - Ftx - AtAd;
%         Mat         = @(y)build_CGMatrix(ind,y);
% 
%         [d(~ind),iterd,flag,~,~,CG_AtAd,relres] = ...
%             mypcgw(Mat,r(~ind),CG_tol,CG_maxit,[],1);
% 
%         nr_CG = nr_CG + iterd;
%         nr_Acall = nr_Acall + iterd*2;
%         xz = x + d;
%         gradxz = grad + AtAd + CG_AtAd;
        
        d = Hlrinv_no(-Ftx);
        xz = x + d;
        gradxz = grad + At(A(d));
        nr_Acall = nr_Acall + 2;
    else
        d = Hlrinv(-Ftx);
        xz = x + d;
        gradxz = grad + At(A(d));
        nr_Acall = nr_Acall + 2;
    end
    [Ftxz, resxz, yxz] = comp_resR(xz,gradxz);
    %[Ftxz, resxz, yxz, objz] = comp_resR(xz,gradxz);
    nrmd = norm(d(:));
    Ftxzd = -idot(Ftxz,d);
    
    if res < 1e-2; rhs = resxz*nrmd^2; else rhs = nrmd^2; end
    %rhs = nrmd^2;
    
    ratio = Ftxzd/rhs;  %ratio = 1;
    resRatiop = resxz/resp;
    
    success = ratio >= eta1; ddflag = 'f';
    if resxz < resFac*max(resnp)
        x = xz; Ftx = Ftxz; res = resxz; ddflag = 'n';
        grad = gradxz; yx = yxz;  %obj = objz;
        resRatiop = res/resp;
        
        resnp(mod(rnpinx,nnewtcomp)+1) = res;
        rnpinx = rnpinx+1;
    elseif success
        x = x + (-Ftxzd/(resxz^2))*Ftxz; %continue;
        [Ftx, res, yx, grad] = comp_res(x);
        ddflag = 'p';
        nr_res = nr_res + 1;
        nr_Acall = nr_Acall + 2;
        resRatiop = res/resp;
    end
    
    f = 0.5*((grad-Atb)'*x + btb) + mu*norm(x,1);
    fdiff = abs(f-fp)/(fp);
    % stopping criteria, res/tau is comparable to SNF.m
    switch stopCriterion
        case 1
            cstop = (mu <= (1+1e-10)*muf) && (res/tau <= tol);
        case 2
            if mu <= (1+1e-10)*muf
                fiter = 0.5*((grad-Atb)'*x + btb) + mu*norm(x,1);
                cstop = ((fiter - fopt)/max(1,abs(fopt)) <= tol);
            else
                cstop = 0;
            end
        case 3
            cstop = (fdiff < tolA) || (res/tau <= 10*tol);
            
    end
    
    if record>=1 && (cstop || ...
            iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit && ~cstop;
            fprintf('\n%s', str_head);
        end
        %fprintf(str_num, iter, mu, res/tau, resRatiop, ...
        %    Ftxzd, rhs, ratio, sig,lambda, flag, iterd, relres, ddflag);
        
        fprintf(str_num, iter, mu, res/tau, resRatiop, ...
            fdiff, rhs, ratio, sig,lambda, flag, relres, ddflag);
    end
    
    
    if cstop
        switch stopCriterion
            case {1,2}
                break;
            case 3
                adjust_mu(grad);
                [Ftx, res, yx, grad] = comp_res(x);
                CG_maxit = CG_maxit + 1;
        end
    end
    
    if ratio >= eta2
        lambda = max(gamma1*lambda, 1e-16);
    elseif ratio >= eta1
        lambda = gamma2*lambda;
    else
        lambda = gamma3*lambda;
    end
    
    if success
        % evaluate the quality of the last iterate
        goodIter = (res <= 0.5*resp)|| (res <= max(0.1*resp_mu, ftol));
        
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
        
        if mu_tau_change; [Ftx, res, yx]  = comp_resR(x,grad); end
        if opts.cont  && mu_tau_change;
            resp_mu = res;
            resnp(mod(rnpinx,nnewtcomp)+1) = res;
            rnpinx = rnpinx+1;
        end
        
        %   update CG parameters if final mu is reached
        if opts.CG_adapt && (mu <= (1+1e-10)*muf)
            if res < res_tol;
                CG_tol     = max(0.08*CG_tol,1e-7);
                res_tol    = 0.1*res_tol;
            end
            
        end
    end
    
    s = d;
    ygk = Ftx - Ftxp;
    
    resp = res; xp = x; Ftxp = Ftx;
    if opts.tau_adapt; gradp = grad; end
    if ~strcmp(ddflag,'f')
        istore = istore + 1;  doLM = 0;
   %     nlbfgs = nlbfgs + 1;
        pos = mod(istore, mm); if pos == 0; pos = mm; end; ppos = pos;
        YK(:,pos) = ygk;  SK(:,pos) = s;

        if istore <= mm
            perm = [perm, pos]; perm2 = perm;
            % STY is S'Y, lower triangular
            STY(istore,1:istore)  = s'*YK(:,1:istore);
            STY(1:istore,istore)  = (ygk'*SK(:,1:istore))';
            % STS is S'S
            STS(1:istore,istore)  = SK(:,1:istore)'*s;
            STS(istore,1:istore)  = STS(1:istore,istore)';
            % YTY is Y'Y
            YTY(1:istore,istore)  = YK(:,1:istore)'*ygk;
            YTY(istore,1:istore)  = YTY(1:istore,istore)';            
        else
            % update YK, SK
            ppos = mm; perm = [perm(mm), perm(1:mm-1)];
            perm2 = [perm2(2:mm), perm2(1)];
            % update STY, STS, first move the old part to upper
            STY(1:end-1, 1:end-1) = STY(2:end, 2:end);
            STS(1:end-1, 1:end-1) = STS(2:end, 2:end);
            YTY(1:end-1, 1:end-1) = YTY(2:end, 2:end);

            % then update the last column or row
            STY(end,perm)  = s'*YK(:,:);
            STY(perm,end)  = (ygk'*SK(:,:))';
            STS(perm,end)  = SK(:,:)'*s;
            STS(end,:)     = STS(:,end)';
            YTY(perm,end)  = YK(:,:)'*ygk;
            YTY(end,:)     = YTY(:,end)';            
        end

        % update BB
        hmu = STY(ppos, ppos)/STS(ppos, ppos);
        DD = diag(diag(STY(1:ppos,1:ppos)));
        LL = tril(STY(1:ppos,1:ppos), -1);
        BB = [hmu*STS(1:ppos,1:ppos), LL; LL', - DD];
    end
    
end %end outer loop

out.time = toc;

out.f = f;
out.iter = iter;
out.res  = res;

out.nr_res = nr_res;
out.nr_CG = nr_CG;
out.Acalls = nr_Acall;% + nr_res*2 + nr_CG*2;

    function adjust_mu(grad)
        temp_mu = max(muf,0.2*max(abs(grad(:))));
        
        if temp_mu > mu
            mu = muf;
        else
            mu = temp_mu;
        end
        
        if mu == muf
            stopCriterion = final_stopCriterion;
            tolA = tol;
        else
            stopCriterion = 3;
            tolA = 1e-3;
        end
    end

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

    function Hinv_x = Hlrinv(sx)
        c1 = sig*(hmu)/(sig+hmu); c2 = hmu/(sig+hmu); c3 = 1/(sig+hmu);
        Rk11 = c1*(STS(1:ppos,1:ppos)); 
        Rk12 = LL-c2*STY(1:ppos,1:ppos); 
        Rk22 = -(DD+c3*(YTY(1:ppos,1:ppos))); 
        Rk = [Rk11,Rk12;Rk12',Rk22];
     %   fprintf('%.2e %.2e %.2e %.2e %.2e %.2e\n',...
     %       cond(Rk11),cond(Rk12),cond(Rk22),cond(Rk),cond(DD),cond(LL))
        Hinv_x = sx/(hmu+sig) + [hmu*SK(:,perm2),YK(:,perm2)]* ...
            (Rk\[hmu*SK(:,perm2)'*sx;YK(:,perm2)'*sx])/(hmu+sig)^2;
    end

    function Hinv_x = Hlrinv_no(sx)
        Hinv_x = sx/(hmu+sig);
    end



end



