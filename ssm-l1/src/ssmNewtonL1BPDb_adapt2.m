function [x, out] = ssmNewtonL1BPDb_adapt2(A,b,opts)

%%-------------------------------------------------------------------------
if nargin < 2; opts = []; end
if ~isfield(opts,'tol');   opts.tol = 1e-6;     end
if ~isfield(opts,'record');opts.record = 1;     end
if ~isfield(opts,'eps');   opts.eps = 1e-4;     end
if ~isfield(opts,'doQN');  opts.doQN = 1;       end
if ~isfield(opts,'doLM');  opts.doLM = 1;       end
if ~isfield(opts,'param'); opts.param = [];     end

if ~isfield(opts,'NTstep'); opts.NTstep  = 10;   end
if ~isfield(opts,'mu');     opts.mu = mean(abs(b));    end
if ~isfield(opts,'delta');  opts.delta = 0; end
if ~isfield(opts,'CG_maxit'); opts.CG_maxit = 5;   end
if ~isfield(opts,'CG_tol');   opts.CG_tol = 1e-1;   end
if ~isfield(opts,'CG_adapt'); opts.CG_adapt = 1;   end

tol    = opts.tol;
record = opts.record;
doQN   = opts.doQN;
doLM   = opts.doLM;

NTstep = opts.NTstep;
mu     = opts.mu;
delta  = opts.delta;
CG_tol = opts.CG_tol;
CG_maxit = opts.CG_maxit;

%%-------------------------------------------------------------------------
% options for fixed-point algorithms
if ~isfield(opts,'sigma'), opts.sigma = 1.618;              end
if ~isfield(opts,'maxits');opts.maxits = 100;           end
if ~isfield(opts,'itPrintFP'); opts.itPrintFP = 1;     end
if ~isfield(opts,'switchTol'); opts.switchTol = 1e-1;   end

sigma    = opts.sigma;
maxits   = opts.maxits;
itPrintFP = opts.itPrintFP;
switchTol = opts.switchTol;

%%-------------------------------------------------------------------------
% options for QN
if ~isfield(opts,'rhols'),  opts.rhols  = 1e-6;     end
if ~isfield(opts,'betals'); opts.betals = 0.5;      end
if ~isfield(opts,'hConst'); opts.hConst = 1e-2;     end
if ~isfield(opts,'mm');     opts.mm  = 5;           end
if ~isfield(opts,'maxNLS'); opts.maxNLS = 10;       end
if ~isfield(opts,'eps');    opts.eps  = 1e-16;      end

rhols    = opts.rhols;
betals   = opts.betals;
hConst   = opts.hConst;
mm       = opts.mm;
maxNLS   = opts.maxNLS;
eps      = opts.eps;



%%-------------------------------------------------------------------------
% options for the Levenberg-Marquardt algorithm
if ~isfield(opts,'maxit');    opts.maxit = 100;   end
if ~isfield(opts,'gtol');     opts.gtol = 1e-6;   end
if ~isfield(opts,'xtol');     opts.xtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts,'muPow');    opts.muPow = 0.7;     end
if ~isfield(opts,'resFac');   opts.resFac = 0.98;   end
if ~isfield(opts,'eta1');     opts.eta1 = 1e-6;   end
if ~isfield(opts,'eta2');     opts.eta2 = 0.9;    end
if ~isfield(opts,'gamma1');   opts.gamma1 = 0.5;  end
if ~isfield(opts,'gamma2');   opts.gamma2 = 1;    end
if ~isfield(opts,'gamma3');   opts.gamma3 = 10;   end
if ~isfield(opts,'lambda');   opts.lambda = 1;    end %adjust mu
if ~isfield(opts,'itPrint');  opts.itPrint = 1;   end
if ~isfield(opts,'maxItStag');opts.maxItStag = 5; end
if ~isfield(opts,'restart');  opts.restart = 1; end
if ~isfield(opts,'maxItPCG'); opts.maxItPCG = 20;   end

maxItStag = opts.maxItStag; restart=opts.restart; maxItPCG=opts.maxItPCG;
maxit   = opts.maxit;   itPrint = opts.itPrint; muPow   = opts.muPow;
gtol    = opts.gtol;    ftol    = opts.ftol;    resFac  = opts.resFac;
eta1    = opts.eta1;    eta2    = opts.eta2;    lambda  = opts.lambda;
gamma1  = opts.gamma1;  gamma2  = opts.gamma2;  gamma3  = opts.gamma3;
%-------------------------------------------------------------------------

%------------------------------------------------------------------
if record >= 1
    % set up record format
    if ispc; str1 = '  %12s'; str2 = '  %8s';
    else     str1 = '  %12s'; str2 = '  %8s'; end
    stra_FP = ['switch to the fixed-point method\n','%5s',str1,str2,'\n'];
    str_head_FP = sprintf(stra_FP, 'iter', 'res', 'resr');
    str_num_FP = ['%4d  %+8.7e  %+2.1e\n'];
    str_num_LBFGS = ['%4d  %+8.7e  %+2.1e   %d   %2.1e   %2.1e   %d\n'];
    stra = ['switch to the Newton method\n',...
        '%5s','%6s', str1, str2,str2, str2, str2, str2, str1, '\n'];
    str_head = sprintf(stra, ...
        'iter', 'mu', 'res', 'resR', 'Ftudd', 'rhs', 'ratio', ...
        'sig', 'CG(flag/it/res)');
    str_num = ['%4d  %+2.1e %+8.7e  %+2.1e  %+2.1e  %+2.1e  %+2.1e ' ...
        '  %+2.1e %1d %2d %2.1e %2s\n'];
end


% define linear operators
[A,At,b,opts] = linear_operators(A,b,opts);
m = length(b);
posdel = isfield(opts,'delta') && delta > 0;
if isfield(opts,'x0'); x = opts.x0; else x = []; end
if isfield(opts,'z0'); z = opts.z0; else z = []; end
if isfield(opts,'u0'); u = opts.u0; else u = []; end

% check zero solution
Atb = At(b);
n = length(Atb);
bnrm = norm(b);
bmax = norm(b,inf);
L2Con_zsol =  posdel && bnrm <= delta;
BP_zsol    = ~posdel && bmax <= tol;
zsol = L2Con_zsol || BP_zsol;
if zsol
    x = zeros(n,1);
    out.iter = 0;
    out.cntAt = 1;
    out.cntA = 0;
    out.exit = 'Data b = 0';
    return;
end
% ========================================================================

out.cntA = 0; out.cntAt = 0;
% scaling data and model parameters
b0 = b;
b  = b0 / bmax;  %mu = mu / bmax;
if posdel, delta = delta / bmax; end
if isfield(opts,'xs'), opts.xs = opts.xs/bmax; end

% if isempty(x); x = At(b); out.cntA = out.cntA + 1;end;
if isempty(u); u = zeros(n,1); end

nrmb = bnrm/bmax;
Au = A(u);
out.cntA = out.cntA + 1;

% mu_orig = mu*10;
% if delta > 0;
%     %mu = mu*0.5; %with noise sigma = 0.1
%     %mu = mu*0.8; %with noise sigma = 0.1
%     mu = mu*0.1;
% else
%     mu = mu*5;  %no noise
% end

% ========================================================================
% % % stage one: fixed-point algorithm
out.iters = 0; res = inf;  out.nfe = 1; avgResRatio = 1;
out.hist.res = [];  out.hist.resIt = [];
if doQN == 1
    FP(switchTol);
    %x = x * bmax; return;
    %[x, u, outs] = yall1_solve(A,At,b,[],[],opts);
    %[Ftz, x, res, data] = comp_res(u);
elseif doQN >= 2
    % set up storage for L-BFGS
    SK = zeros(n,mm);% S stores the last ml changes in x
    YK = zeros(n,mm); % Y stores the last ml changes in gradient of f.
    YTY = zeros(mm);
    STY = zeros(mm);
    rho = zeros(mm,1);
    istore = 0; pos = 0;  ppos = 0; status = 0;  perm = []; ygk = [];
    gam = 1;
    EQSOLVE(switchTol);
else
    [Ftz, x, res, data] = comp_res_fp(u);
    %xp = x;
end

%
% optf = optimoptions('fsolve','Algorithm','levenberg-marquardt',...
%     'Display','iter','Jacobian','off', 'TolX', 1e-12, 'TolFun', 1e-12, ...
%     'DerivativeCheck','on');
% [x0,fval,exitflag,out3] = fsolve(@comp_res, u(:),optf);
% out.iter = out3.iterations;
% exitflag
% out3
%
% x = sign(x0).*max(abs(x0)-mu,0);
% x = x * bmax;
% return


resp = res; resp_mu = res;
reshist = res; maxNTres = res;

% stage two: Newton method
cstop = 0; % stop

mu_update_itr = 5;
mu_delta = 10/bmax;
mu_fact = 3; 
mu_min = 1e-6;
mu_max = 1e6;
smean = @geo_mean;
for iter = 1:maxit
    %----------------------------------------------------------------------
    % parameter for the revised Newton system
    sig = lambda*(resp^muPow);
    %sig = 1d-5*resp;
    
    pcgTol = max(min(0.1*resp, 0.1),gtol);
    %pcgTol = 1e-3; CG_maxit = 20;
    
    %[d, flag, relres, iterd] = qmr(@aJact,-Ftz,pcgTol,CG_maxit);
    %[d, flag, relres, iterd] = bicgstab(@aJac,-Ftz,pcgTol, CG_maxit);
    %[d, flag, relres, iterd] = bicgstabl(@aJac,-Ftz,pcgTol, CG_maxit);
    %iterd = round(iterd);
    %[d, flag, relres, iterd] = tfqmr(@aJac,-Ftz,pcgTol, CG_maxit);
    %[d,iterd,flag,~,~,~,relres] = mypcgw(@aJac,-Ftz,pcgTol,CG_maxit,[],0);
    
    iH = ones(n,1); iH(data.Ixmzp) = 1/sig; iH(data.Ixmzn) = 1/(1+sig);
    WW = ones(n,1); WW(data.Ixmzp) = -1;  iHWW = iH.*WW;
    d = -iH.*Ftz;
    if delta > 0
        if data.nrmAub > delta
            Bmat = @(z) (1-delta/data.nrmAub)*z + ...
                (delta*(z'*data.Aub)/(data.nrmAub).^3)*data.Aub;
            aaa = Bmat(A(WW.*d));
            [ds, flag, relres, iterd] = bicgstabl(@JJmat,aaa,pcgTol, CG_maxit);
            d = d + iH.*At(ds);
            out.cntA  = out.cntA  + 1;
            out.cntAt = out.cntAt + 1;
        end
    else
        aaa = A(WW.*d);
        %[ds, flag, relres, iterd] = bicgstabl(@JJmat0,aaa,pcgTol, CG_maxit);
        [ds,iterd,flag,~,~,~,relres] = mypcgw(@JJmat0,aaa,pcgTol,CG_maxit,[],0);
        d = d + iH.*At(ds);
        out.cntA  = out.cntA  + 1;
        out.cntAt = out.cntAt + 1;
    end
    iterd = round(iterd);
    
    %----------------------------------------------------------------------
    
    ud = u + d;
    Aup = Au;
    Au = A(ud);
    out.cntA  = out.cntA  + 1;
    [Ftud, xud, resud, dataud] = comp_res_fp(ud);
    nrmd = norm(d(:));
    Ftudd = -idot(Ftud,d);
    
    if res < 1e-3; rhs = resud*nrmd^2; else rhs = nrmd^2; end
    
    ratio = Ftudd/rhs;  %ratio = 1;
    resRatiop = resud/resp;  resRatiop1 = resRatiop;
    
    success = ratio >= eta1; ddflag = 'f';
    if success
%         if res < 3e-3
%             mu_delta = 0.8;
%         end
        nhist = min(NTstep,length(reshist));
        maxNTres = max(reshist(end-nhist+1:end));
        %if resud > resFac*resp
        if resud > resFac*maxNTres;
            u = u + (-Ftudd/resud^2)*Ftud; %continue;
            Au = Aup +  (-Ftudd/resud^2)*dataud.AFtz;
            ddflag = 'p';
            [Ftz, x, res, data] = comp_res_fp(u);
        else
            u = ud; Ftz = Ftud; res = resud; ddflag = 'n';
            data = dataud;  x = xud;
        end
        
        % check relative change
        %rdnrm = norm(x - xp);
        %xrel_chg = rdnrm/max(1,norm(xp));
        % stopping criteria
        cstop = (res <= tol); % || xrel_chg <= tol;
        
        resRatiop = res/resp;
        resp = res;  %xp = x;
        out.hist.res = [out.hist.res; res];
    end
    
    if record>=1 && (cstop || ...
            iter == 1 || iter==maxit || mod(iter,itPrint)==0)
        if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit && ~cstop;
            fprintf('\n%s', str_head);
        end
%         fprintf(str_num, iter, mu, res, resRatiop, ...
%            Ftudd, rhs, ratio, sig, flag, iterd, relres, ddflag);
%         fprintf(str_num, iter, mu, res, resRatiop, ...
%             resRatiop1, rhs, ratio, sig, flag, iterd, relres, ddflag);
       if iter > 1
           fprintf('%d res: %2.1e %2.8e, %2.8e %2.8e %2s %2.1e %2.1e,%2.1e,%2.1e\n',...
               iter,res, pinf, dinf,norm(x,1),ddflag,resRatiop,mu_delta,mu_fact,norm(u));
       end
    end
    
    if cstop
        out.msg = 'optimal';
        break;
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
        goodIter =  (res <= 0.5*resp) || (res <= max(0.5*resp_mu, 10*tol));
        
        %         % continuation: update mu
        %         if goodIter
        %             %mfrac = 0.1; big = 50; nup = 8;
        %             %mu_min = mfrac^nup * mu_orig;
        %             %muredf = max((1-(0.65^log10(mu0/mu))*0.535),0.15);
        %             mup = mu;
        %             mu = min(2.5*mu,mu_orig);
        %             mu_change = 0; if abs(mu-mup) > 1e-12; mu_change = 1; end
        %
        %             if mu_change;
        %                 [Ftz, x, res, data] = comp_res(u);
        %                 resp_mu = res;
        %                 resp = res;  xp = x;
        %             end
        %         end
        
        % update CG parameters if final mu is reached
        mu_change = 0;
        if res < 1e-9;
            CG_maxit   = 10;
            muPow      = 0.15;
        elseif res < 1e-5;
            CG_maxit   = 6;
            muPow      = 0.6;
        elseif res < 1e-4;
            CG_maxit   = 5;
            muPow      = 0.6;
        elseif res < 1e-3;
            CG_maxit   = 4;
            muPow      = 0.6;
        elseif res < 1e-1;
            CG_maxit   = 2;
            muPow      = 0.6;
        elseif res < 1e0
            CG_maxit   = 2;
            muPow      = 0.7;
        else
            CG_maxit   = 2;
            muPow      = 0.7;
        end
        
    end
    
    Ax = data.Ax;
    tempu = u+sigma*Ftz;
    axmz = abs(tempu);
    xp = sign(tempu).*max(axmz-mu,0);
    rpnrm = norm(Ax-b);
    rdnrm = norm(x-xp);
    pinf = rpnrm/nrmb;
    dinf = rdnrm/norm(x-u);

    %hist.pobj(iters) = objp;
    %hist.dobj(iters) = objd;
    %hist.gap(iters)  = pdgap;
    hists.pinf(iter) = pinf;
    hists.dinf(iter) = dinf;

    hists.pvd(iter)  = pinf/dinf;
    hists.dvp(iter)  = dinf/pinf;
    
    if mod(iter,mu_update_itr)==0
        mup = mu;
        sitr = iter-mu_update_itr+1;
        avg_pvd = smean(hists.pvd(sitr:iter));
        avg_dvp = smean(hists.dvp(sitr:iter));
        if avg_pvd > mu_delta 
          mu = mu/(mu_fact);
        elseif avg_dvp > mu_delta 
          mu = mu*(mu_fact);
        end
% %         if avg_dvp > mu_delta
% %           mu = mu*(mu_fact);
% %         else
% %           mu = mu/(mu_fact);
% %         end
        mu = max(min(mu,mu_max),mu_min);
        %               rdmu = rho / mu; rdmu1 = rdmu + 1;
        %               bdmu = b / mu; ddmu = delta / mu;
        if record >=1; fprintf('  -- mu updated: %f\n',mu); end

        if mu ~= mup
            [Ftz, x, u, res, data] = comp_res_fp_updmu(u,x,data,mu,mup);
        end
   end

    
    
end %end outer loop

out.iter = iter;
out.res  = res;

% restore solution x
x = x * bmax;


    function a = idot(x,y)
        a = real(x(:)'*y(:));
    end

    function [Ftz, x, res, data] = comp_res(u)
        
        axmz = abs(u);
        data.Ixmzp = axmz >= mu; data.Ixmzn = axmz < mu;
        x = sign(u).*max(axmz-mu,0);
        
        xu = 2*x - u;
        Aub = A(xu) - b;
        if delta > 0
            nrmAub = norm(Aub(:)); data.nrmAub = nrmAub;
            if nrmAub > delta;
                AtAub = At(Aub);
                xu = xu + AtAub*(-1+delta/nrmAub);
                data.Aub = Aub;
                out.cntA  = out.cntA  + 1;
                out.cntAt = out.cntAt + 1;
            else
                out.cntA  = out.cntA  + 1;
            end
        else
            xu = xu - At(Aub);
            out.cntA  = out.cntA  + 1;
            out.cntAt = out.cntAt + 1;
        end
        
        Ftz = x - xu;
        res = norm(Ftz);
    end
    

    function [Ftz, x, res, data] = comp_resR(u, x, Ftz)
        axmz = abs(Ftz);
        data.Ixmzp = axmz >= mu; data.Ixmzn = axmz < mu;
        Ftz = sign(Ftz).*max(axmz-mu,0);
        Ftz = x - Ftz;
        res = norm(Ftz);
    end


    function Js = JJmat(sx, data)
        Js = sx - Bmat(A(iHWW.*At(sx)));
        out.cntA  = out.cntA  + 1;
        out.cntAt = out.cntAt + 1;
    end

    function Js = JJmat0(sx, data)
        Js = sx - (A(iHWW.*At(sx)));
        out.cntA  = out.cntA  + 1;
        out.cntAt = out.cntAt + 1;
    end


    function Js = comp_Jt(sx,data)
        if delta > 0
            if data.nrmAub > delta
                Asx = A(sx);
                Asx = (1 - (delta/data.nrmAub))*Asx + ...
                    (delta*(Asx'*data.Aub)/(data.nrmAub)^3)*data.Aub;
                Js = sx - At(Asx);
                %sx'*Js
                out.cntA  = out.cntA  + 1;
                out.cntAt = out.cntAt + 1;
            else
                Js = sx;
            end
        else
            Js = sx - At(A(sx));
            out.cntA  = out.cntA  + 1;
            out.cntAt = out.cntAt + 1;
        end
        Js(data.Ixmzp) = -Js(data.Ixmzp);
        sx(data.Ixmzn) = 0;
        Js = Js + sx;
    end


    function Jts = comp_J(sx,data)
        Jts = sx;
        Jts(data.Ixmzp) = -Jts(data.Ixmzp);
        if delta > 0
            if data.nrmAub > delta
                Asx = A(Jts);
                Asx = (1 - (delta/data.nrmAub))*Asx + ...
                    (delta*(Asx'*data.Aub)/(data.nrmAub)^3)*data.Aub;
                Jts = Jts - At(Asx);
            end
        else
            Jts = Jts - At(A(Jts));
        end
        sx(data.Ixmzn) = 0;
        Jts = Jts + sx;
        out.cntA  = out.cntA  + 1;
        out.cntAt = out.cntAt + 1;
    end


    function yy = aJac(xx)
        yy = comp_J(xx,data) + sig*xx;
    end

    function yy = aJact(xx,transp_flag)
        if strcmp(transp_flag,'transp')      % y = A'*x
            yy = comp_Jt(xx,data) + sig*xx ;
        elseif strcmp(transp_flag,'notransp') % y = A*x
            yy = comp_J(xx,data) + sig*xx;
        end
    end

    function [Ftz, x, res, data] = comp_res_fp(u)
        axmz = abs(u);
        data.Ixmzp = axmz >= mu; data.Ixmzn = axmz < mu;
        x = sign(u).*max(axmz-mu,0);
        Ax = A(x);
        xu = 2*x - u;
        Axu = 2*Ax - Au;
        Aub = Axu-b;
        %Aub = A(xu) - b;
        if delta > 0
            nrmAub = norm(Aub(:)); data.nrmAub = nrmAub;
            if nrmAub > delta;
                AtAub = At(Aub);
                xu = xu + AtAub*(-1+delta/nrmAub);
                Axu = Axu + Aub*(-1+delta/nrmAub);
                data.Aub = Aub;
                out.cntA  = out.cntA  + 1;
                out.cntAt = out.cntAt + 1;
            else
                out.cntA  = out.cntA  + 1;
            end
        else
            xu = xu - At(Aub);
            Axu = Axu - Aub;
            out.cntA  = out.cntA  + 1;
            out.cntAt = out.cntAt + 1;
        end

        Ftz = x - xu;
        AFtz = Ax - Axu;
        data.AFtz = AFtz;
        data.Ax = Ax;
        res = norm(Ftz);
    end
    function [Ftz, x,u, res, data] = comp_res_fp_updmu(u,x,data,mu,mup)
        Ax = data.Ax;
        xu = (1+mu/mup)*x - mu/mup*u;
        Axu = (1+mu/mup)*Ax - mu/mup*Au;
        Aub = Axu-b;
        %Aub = A(xu) - b;
        if delta > 0
            nrmAub = norm(Aub(:)); data.nrmAub = nrmAub;
            if nrmAub > delta;
                AtAub = At(Aub);
                xu = xu + AtAub*(-1+delta/nrmAub);
                Axu = Axu + Aub*(-1+delta/nrmAub);
                data.Aub = Aub;
                out.cntAt = out.cntAt + 1;
            end
        else
            xu = xu - At(Aub);
            Axu = Axu - Aub;
            out.cntAt = out.cntAt + 1;
        end

        tempxu = xu + mu/mup*(u - x);
        axmz = abs(tempxu);
        u = mu/mup*(u - x) + xu;
        Au = mu/mup*(Au - Ax) + Axu;
        x = sign(tempxu).*max(axmz-mu,0);
        Ax = A(x);

        xu = 2*x - u;
        Axu = 2*Ax - Au;
        Aub = Axu-b;
        %Aub = A(xu) - b;
        if delta > 0
            nrmAub = norm(Aub(:)); data.nrmAub = nrmAub;
            if nrmAub > delta;
                AtAub = At(Aub);
                xu = xu + AtAub*(-1+delta/nrmAub);
                Axu = Axu + Aub*(-1+delta/nrmAub);
                data.Aub = Aub;
                out.cntA  = out.cntA  + 1;
                out.cntAt = out.cntAt + 1;
            else
                out.cntA  = out.cntA  + 1;
            end
        else
            xu = xu - At(Aub);
            Axu = Axu - Aub;
            out.cntA  = out.cntA  + 1;
            out.cntAt = out.cntAt + 1;
        end

        Ftz = x - xu;
        AFtz = Ax - Axu;
        data.AFtz = AFtz;
        data.Ax = Ax;
        res = norm(Ftz);
    end

    function FP(switchTol)
        out.hist.resIt = [out.hist.resIt; length(out.hist.res)];
        stol = max(tol,switchTol); reshist = zeros(maxits,1);
        Au = A(u); xp = sign(u).*max(abs(u)-mu,0);
        [Ftz, x, res, data] = comp_res_fp(u);
        sigma = 1;
        u = u - sigma*Ftz;
        xp = x;
        Au = Au - sigma*data.AFtz;
        
        mu_update_itr = 15;
        mu_delta = 10/bmax;
        mu_fact = 1/0.75; 
        mu_min = 1e-6;
        mu_max = 1e6;
        smean = @geo_mean;
        for iters = 1:maxits
            rest = res;
            [Ftz, x, res, data] = comp_res_fp(u);
            Ax = data.Ax;
            reshist(iters) = res/rest;
            out.hist.res = [out.hist.res; res];
            cstop = res < stol;
            
            % check optimality
            %check_opti;
            %if print > 1; iprint2; end
            
            
            rpnrm = norm(Ax-b);
            rdnrm = norm(x-xp);
            pinf = rpnrm/nrmb;
            dinf = rdnrm/norm(x-u);

            %hist.pobj(iters) = objp;
            %hist.dobj(iters) = objd;
            %hist.gap(iters)  = pdgap;
            hist.pinf(iters) = pinf;
            hist.dinf(iters) = dinf;

            hist.pvd(iters)  = pinf/dinf;
            hist.dvp(iters)  = dinf/pinf;

            %  check_stopping;
            %ainf = max([pinf,dinf,gap]); cstop = ainf <= tol;
            ainf = max([pinf,dinf]); %cstop = ainf <= tol;
            
            if record>=1 && (cstop || ...
                    iters == 1 || iters==maxits || mod(iters,itPrintFP)==0)
                if iters == 1 || (mod(iters, 20*itPrintFP) == 0 && ...
                        iters ~= maxits && ~cstop);
                    fprintf('\n%s', str_head_FP);
                end
                %fprintf(str_num_FP,iters, res, reshist(iters));
                fprintf('%d res: %2.1e %2.8e, %2.8e %2.8e\n',iters,res, pinf, dinf,norm(x,1));
            end

            %% other chores
            if cstop; break; end 
            
            if (iters > 1 && cstop) || iters == maxits; break; end
            %if iters == 1; stol = min(switchTol, 0.1*res); end
            
            %if adp_mu == 1 
              update_mu_drs; 
            %end
            
            u = u - sigma*Ftz;
            xp = x;
            Au = Au - sigma*data.AFtz;
            
        end
        avgResRatio = median([1;reshist(2:iters)]);
        out.iters = out.iters + iters;
        if record >= 1;
            fprintf('averaged res ratio: %2.1e\n', avgResRatio);
        end
        out.hist.resIt = [out.hist.resIt; length(out.hist.res)];
        
        
        
        % nested function
        function update_mu_drs    
            if mod(iters,mu_update_itr)==0
                mup = mu;
                sitr = iters-mu_update_itr+1;
                avg_pvd = smean(hist.pvd(sitr:iters));
                avg_dvp = smean(hist.dvp(sitr:iters));
                if avg_pvd > mu_delta 
                  mu = mu/(mu_fact);
                elseif avg_dvp > mu_delta 
                  mu = mu*(mu_fact);
                end
                mu = max(min(mu,mu_max),mu_min);
                %               rdmu = rho / mu; rdmu1 = rdmu + 1;
                %               bdmu = b / mu; ddmu = delta / mu;
                if record >=1; fprintf('  -- mu updated: %f\n',mu); end
                
                if mu ~= mup
                    [Ftz, x, u, res, data] = comp_res_fp_updmu(u,x,data,mu,mup);
                end
           end
        end
    end


% z is x in Donghui Li's paper
% -Ftz is firmly nonexpansive
    function EQSOLVE(switchTol)
        stol = max(tol,switchTol); reshist = zeros(maxits,1);
        [Ftz, x, res, data] = comp_res(u);
        resp = res;   success = 1;
        out.hist.resIt = [out.hist.resIt; length(out.hist.res)];
        for iters = 1:maxits
            Ftzp = Ftz;
            if istore == 0; d = -Ftz;
            else            d = LBFGS_Hg_Loop(-Ftz); end
            alp = 1;
            nrmd = norm(d(:));
            for dls = 0:maxNLS
                ud = u + alp*d;
                [Ftud, xud, resud, dataud] = comp_res(ud);
                Ftxzd = -idot(Ftud,d);
                rhs = rhols*alp*resud*nrmd^2;
                if record >= 2
                    fprintf('ls: %2d, Ftxzd: %2.1e, rhs: %2.1e\n', ...
                        dls, Ftxzd, rhs);
                end
                if Ftxzd >= rhs
                    break;
                end
                alp = betals*alp;
            end
            
            rest = res;
            if resud > resFac*resp
                u = u + (-alp*Ftxzd/resud^2)*Ftud; ddflag = 'p';
                [Ftz, x, res, data] = comp_res(u);
            else
                u = ud; Ftz = Ftud; res = resud; ddflag = 'n';
                data = dataud;  x = xud;
            end
            
            reshist(iters) = res/rest;
            out.hist.res = [out.hist.res; res];
            cstop = res < stol || resud < stol;
            if record>=1 && (cstop || ...
                    iters == 1 || iters==maxits || mod(iters,itPrintFP)==0)
                if iters == 1 || (mod(iters, 20*itPrintFP) == 0 && ...
                        iters ~= maxits && ~cstop);
                    fprintf('\n%s', str_head_FP);
                end
                fprintf(str_num_LBFGS,iters, res, reshist(iters),...
                    success,resud, alp, dls);
            end
            
            if iters > 1 && cstop; break; end
            if iters == 1; stol = min(switchTol, 0.1*res); end
            
            %----------------------------------------------------------------------
            % save for L-BFGS
            ygk = Ftud - Ftzp; d = alp*d; nrmd = alp*nrmd;
            ygk = ygk + (hConst*res^0)*d;
            stygk = idot(d,ygk);
            success = stygk>eps*(nrmd)^2;
            %Check to save s and y for L-BFGS.
            if success && doQN > 1
                istore = istore + 1;
                pos = mod(istore, mm); if pos == 0; pos = mm; end;
                YK(:,pos) = ygk;  SK(:,pos) = d;   rho(pos) = 1/stygk;
                
                if istore <= mm; status = istore; perm = [perm, pos];
                else status = mm; perm = [perm(2:mm), perm(1)]; end
            end
        end
        avgResRatio = median([1;reshist(2:iters)]);
        out.iters = out.iters + iters;
        if record >= 1;
            fprintf('averaged res ratio: %2.1e\n', avgResRatio);
        end
        out.hist.resIt = [out.hist.resIt; length(out.hist.res)];
    end

% computer y = H*v where H is L-BFGS matrix
    function y = LBFGS_Hg_Loop(dv)
        q = dv;   alpha = zeros(status,1);
        for di = status:-1:1;
            k = perm(di);
            alpha(di) = (q'*SK(:,k)) * rho(k);
            q = q - alpha(di)*YK(:,k);
        end
        y = q/(rho(pos)* (ygk'*ygk));
        for di = 1:status
            k = perm(di);
            beta = rho(k)* (y'* YK(:,k));
            y = y + SK(:,k)*(alpha(di)-beta);
        end
    end
end



function a = idot(x,y)
a = real(x(:)'*y(:));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A,At,b,opts] = linear_operators(A0, b0, opts)
%
% define linear operators A and At
% (possibly modify RHS b if nu > 0)
%
b = b0;
if isnumeric(A0);
    if size(A0,1) > size(A0,2);
        error('A must have m < n');
    end
    A  = @(x) A0*x;
    At = @(y) (y'*A0)';
elseif isstruct(A0) && isfield(A0,'times') && isfield(A0,'trans');
    A  = A0.times;
    At = A0.trans;
elseif isa(A0,'function_handle')
    A  = @(x) A0(x,1);
    At = @(x) A0(x,2);
else
    error('A must be a matrix, a struct or a function handle');
end

% use sparsfying basis W
if isfield(opts,'basis')
    C = A; Ct = At; clear A At;
    B  = opts.basis.times;
    Bt = opts.basis.trans;
    A  = @(x) C(Bt(x));
    At = @(y) B(Ct(y));
end

% solving L1-L1 model if nu > 0
if isfield(opts,'nu') && opts.nu > 0
    C = A; Ct = At; clear A At;
    m = length(b0);
    nu = opts.nu;
    t = 1/sqrt(1 + nu^2);
    A  = @(x) ( C(x(1:end-m)) + nu*x(end-m+1:end) )*t;
    At = @(y) [ Ct(y);  nu*y ]*t;
    b = b0*t;
end

if ~isfield(opts,'nonorth');
    opts.nonorth = check_orth(A,At,b);
    if opts.nonorth; error('the rows of A are not orthonormal'); end
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nonorth = check_orth(A, At, b)
%
% check whether the rows of A are orthonormal
%
nonorth = 0;
s1 = randn(size(b));
s2 = A(At(s1));
err = norm(s1-s2)/norm(s1);
if err > 1.e-12; nonorth = 1; end
end



function [x, z, out] = yall1(A, b, opts)
% define linear operators
[A,At,b,opts] = linear_operators(A,b,opts);

m = length(b);
L1L1 = isfield(opts,'nu') && opts.nu > 0;
if L1L1 && isfield(opts,'weights')
    opts.weights = [opts.weights(:); ones(m,1)];
end

% parse options
posrho = isfield(opts,'rho')    && opts.rho > 0;
posdel = isfield(opts,'delta')  && opts.delta > 0;
posnu  = isfield(opts,'nu')     && opts.nu > 0;
nonneg = isfield(opts,'nonneg') && opts.nonneg == 1;
if isfield(opts,'x0'); x0 = opts.x0; else x0 = []; end
if isfield(opts,'z0'); z0 = opts.z0; else z0 = []; end

% check conflicts % modified by Junfeng
if posdel && posrho || posdel && posnu || posrho && posnu
    fprintf('Model parameter conflict! YALL1: set delta = 0 && nu = 0;\n');
    opts.delta = 0; posdel = false;
    opts.nu    = 0; posnu  = false;
end
prob = 'the basis pursuit problem';
if posrho, prob = 'the unconstrained L1L2 problem'; end
if posdel, prob = 'the constrained L1L2 problem';   end
if posnu,  prob = 'the unconstrained L1L1 problem'; end
%disp(['YALL1 is solving ', prob, '.']);

% check zero solution % modified by Junfeng
Atb = At(b);
bmax = norm(b,inf);
L2Unc_zsol = posrho && norm(Atb,inf) <= opts.rho;
L2Con_zsol = posdel && norm(b) <= opts.delta;
L1L1_zsol  = posnu  && bmax < opts.tol;
BP_zsol    = ~posrho && ~posdel && ~posnu && bmax < opts.tol;
zsol = L2Unc_zsol || L2Con_zsol || BP_zsol || L1L1_zsol;
if zsol
    n = length(Atb);
    x = zeros(n,1);
    out.iter = 0;
    out.cntAt = 1;
    out.cntA = 0;
    out.exit = 'Data b = 0';
    return;
end
% ========================================================================

% scaling data and model parameters
b1 = b / bmax;
if posrho, opts.rho   = opts.rho / bmax; end
if posdel, opts.delta = opts.delta / bmax; end
if isfield(opts,'xs'), opts.xs = opts.xs/bmax; end

% solve the problem
t0 = cputime;
[x1,out] = yall1_solve(A, At, b1, x0, z0, opts);
out.cputime = cputime - t0;

% restore solution x
x = x1 * bmax;
if L1L1; x = x(1:end-m); end
if isfield(opts,'basis')
    x = opts.basis.trans(x);
end
if nonneg; x = max(0,x); end

end



