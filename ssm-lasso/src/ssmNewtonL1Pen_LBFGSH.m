function [x, out] = ssmNewtonL1Pen_LBFGS(A,At,b,N,mu,opts)

tic;
%%-------------------------------------------------------------------------
if nargin < 5; opts = []; end
if ~isfield(opts,'x0');    opts.x0  = zeros(N,1);   end
if ~isfield(opts,'tol');   opts.tol = 1e-6;     end
if ~isfield(opts,'record');opts.record = 1;     end
if ~isfield(opts,'eps');   opts.eps = 1e-16;     end

if ~isfield(opts,'comp_obj'); opts.comp_obj = 0;   end
if ~isfield(opts,'tau');      opts.tau = 6;   end
if ~isfield(opts,'tau_adapt');opts.tau_adapt = 0;   end
if ~isfield(opts,'tau_m');    opts.tau_m = 1e-3;   end
if ~isfield(opts,'tau_M');    opts.tau_M = 1e+4;   end
if ~isfield(opts,'maxitTau'); opts.maxitTau = 20;   end
if ~isfield(opts,'teta');     opts.teta = 0.85;   end
if ~isfield(opts,'beta');     opts.beta = 1e-1;   end
if ~isfield(opts,'gamma');    opts.gamma = 1e-1;   end
if ~isfield(opts,'CG_maxit'); opts.CG_maxit = 5;   end
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
% options for QN
if ~isfield(opts,'mm');     opts.mm   = 2;        end
if ~isfield(opts,'maxNLS'); opts.maxNLS = 0;       end
if ~isfield(opts,'rhols'),  opts.rhols  = 1e-6;     end
if ~isfield(opts,'betals'); opts.betals = 0.5;      end
if ~isfield(opts,'hConst'); opts.hConst = 1e-2;     end
mm = opts.mm;
rhols    = opts.rhols;
betals   = opts.betals;
hConst   = opts.hConst;
maxNLS   = opts.maxNLS;

n = length(x);

% set up storage for L-BFGS
SK = zeros(n,mm);	% S stores the last ml changes in x
YK = zeros(n,mm);	% Y stores the last ml changes in gradient of f.
istore = 0; pos = 0;  status = 0;  perm = []; flag = 1; relres = 0;
hmu = 1;

%------------------------------------------------------------------
if record >= 1
    % set up print format
    if ispc; str1 = '  %12s'; str2 = '  %8s';
    else     str1 = '  %12s'; str2 = '  %8s'; end
    stra = ['switch to the LBFGS method\n',...
        '%5s','%6s', str1, str2,str2, str2, str2, '\n'];
    str_head_LBFGS = sprintf(stra, ...
        'iter', 'mu', 'res', 'resR', 'Ftxzd', 'rhs', 'tau');
    str_num_LBFGS = ['%4d %+2.1e %+8.7e  %2.1e %+2.1e %2.1e %2.1e %2s\n'];
    stra = ['switch to the Newton method\n',...
        '%5s','%6s', str1, str2,str2, str2, str2, str2, str2, str1, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'mu', 'res', 'resR', 'Ftxzd', 'rhs', 'ratio', ...
        'sig','lambda','CG(flag/it/res)');
    str_num = ['%4d  %+2.1e %+8.7e  %+2.1e  %+2.1e  %+2.1e  %+2.1e  %+2.1e ' ...
        '  %+2.1e %1d %2.1e %2s\n'];
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
xp = x;  resp = res; Ftxp = Ftx;
resnp(mod(rnpinx,nnewtcomp)+1) = res;
rnpinx = rnpinx+1;
if opts.tau_adapt; gradp = grad; end
if opts.cont; resp_mu = res; end

% stage two: Levenberg-Marquardt algorithm
% itStag = 0; iterd = 1; flag = []; relres = [];
nr_CG = 0;
nr_res = 1;
nr_Acall = 3;

itTau = 0;
for iter = 1:maxit
    itTau = itTau + 1;
    xp = x; Ftxp = Ftx; resp = res; 
    %----------------------------------------------------------------------
    if iter == 1
        d = -Ftx;
    else
        %d = LBFGS_Hg_Loop(-Ftx);
        %d = zeros(n,1);
        lbfgsUpdate(d,-Ftx,status,perm,rhomm,HDiag0,SK,YK);
        
%         d2 = zeros(n,1);
%         lbfgsUpdate(d2,-Ftx,status,perm,rhomm,HDiag0,SK,YK);
%         norm(d2-d,1)
    end
       
    nrmd = norm(d(:));
    AtAd = At(A(d));      nr_Acall = nr_Acall + 2;
    stp = 1; ddflag = 'f'; success = 1;
    for dls = 0:maxNLS
        xz = x + stp*d;
        gradxz = grad + stp*AtAd;
        [Ftxz, resxz, yxz] = comp_resR(xz,gradxz);
        Ftxzd = -idot(Ftxz,d);
        rhs = rhols*stp*resxz*nrmd^2;
        if record >= 2
            fprintf('ls: %2d, Ftxzd: %2.1e, rhs: %2.1e\n', ...
                dls, Ftxzd, rhs);
        end
        
        % good if residual is reduced
        if resxz < resFac*max(resnp) 
            break;
        elseif Ftxzd >= rhs
            success = 0;
            break;
        end
        stp = betals*stp;
    end
    
    if success
        x = xz; Ftx = Ftxz; res = resxz; ddflag = 'n';
        grad = gradxz; yx = yxz;  %obj = objz;
        resRatiop = res/resp;        
        resnp(mod(rnpinx,nnewtcomp)+1) = res;
        rnpinx = rnpinx+1;
    else
        x = x + (-Ftxzd/(resxz^2))*Ftxz; %continue;
        [Ftx, res, yx, grad] = comp_res(x);
        ddflag = 'p';
        nr_res = nr_res + 1;
        nr_Acall = nr_Acall + 2;
        resRatiop = res/resp;
    end
    
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
            fprintf('\n%s', str_head_LBFGS);
        end
        fprintf(str_num_LBFGS, iter, mu, res/tau, resRatiop, ...
            Ftxzd, rhs, tau, ddflag);
    end
    
    
    if cstop
        if opts.crit == 2
            out.f = fiter;
        end
        out.msg = 'optimal';
        break;
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
        if opts.cont  && mu_tau_change;
            resp_mu = res;
            resnp(mod(rnpinx,nnewtcomp)+1) = res;
            rnpinx = rnpinx+1;
        end
    end
    if opts.tau_adapt; gradp = grad; end
    
%     s = xz - xp;
%     if strcmp(ddflag,'n')
%         ygk = Ftx - Ftxp;
%     else
%         ygk = Ftxz - Ftxp + h*Ftxp^r*s;
%     end
    s = x - xp;
    ygk = Ftx - Ftxp;
    stygk = idot(s,ygk);
    %update = stygk>eps*(nrmd)^2;
    if ~strcmp(ddflag,'f')
        istore = istore + 1;
        pos = mod(istore, mm); if pos == 0; pos = mm; end;
        YK(:,pos) = ygk;  SK(:,pos) = s;   rhomm(pos) = 1/stygk;
        
        if istore <= mm; status = istore; perm = [perm, pos];
        else status = mm; perm = [perm(2:mm), perm(1)]; end
        
        HDiag0 = 1/((rhomm(pos))* (ygk'*ygk) );
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

   %lbfgsUpdate(yy,dv,status,perm,rhomm,HDiag0,SK,YK)
% computer y = H*v where H is L-BFGS matrix
    function yy = LBFGS_Hg_Loop(dv)
        yy = dv;   alp = zeros(status,1);
        for di = status:-1:1;
            kk = perm(di);
            alp(di) = (yy'*SK(:,kk)) * rhomm(kk);
            yy = yy - alp(di)*YK(:,kk);
        end
        %yy = qq/(sqrt((ygk'*ygk)/(s'*s)));
        yy = HDiag0*yy;
        for di = 1:status
            kk = perm(di);
            betaa = rhomm(kk)* (yy'* YK(:,kk));
            yy = yy + SK(:,kk)*(alp(di)-betaa);
        end
    end
end



