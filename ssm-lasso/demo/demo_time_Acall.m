%clear all
 
addpath(genpath('../../ssm-lasso'));

test_times = 17;
dyna = 20;
%tol     = [1e-0,1e-1,1e-2,1e-4,1e-6];
tol     = [1e-6];


% % signal size
% N       = 512^2;
% % number of measurements
% M       = floor(N/8);
% % number of nonzeros
% K       = floor(M/5);

% signal size
N       = 256;
% number of measurements
M       = floor(0.5*N);
% number of nonzeros
K       = floor(0.5*M);

% noiselevel
sigma   = 0.5;
% options setting
setting = 'old';

% transformations
U       = @(y) dct(y);
Ut      = @(y) idct(y);


switch dyna
    case 20
        data    = load('data_dyna_20.mat','seed','mu');
    case 40
        data    = load('data_dyna_40.mat','seed','mu');
    case 60
        data    = load('data_dyna_60.mat','seed','mu');
    case 80
        data    = load('data_dyna_80.mat','seed','mu');    
    otherwise
        error('data not found!');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nme = 8;
time = zeros(nme,length(tol));
Acount = zeros(nme,length(tol));


opts1 = [];
opts2 = [];
opts3 = [];
%for j = 1:length(data.mu)
for j = 1:1  
    mu = data.mu(j);
    mu = 1e-2;
    [~,b,~,R,Rt]    = createSignal(N,M,K,dyna,sigma,U,data.seed(j));
    A               = @(y) R(U(y));
    At              = @(y) Ut(Rt(y));

    % optimality measure
    gradx  = @(x) x-At(A(x)-b);
    resfun = @(x,xgrad) norm(x - sign(xgrad).*max(abs(xgrad)-mu,0),'fro');

    x0 = zeros(N,1);

    %-----------------------------------------
    switchTol = 1e-4;
    opts2.switchTol = switchTol;
    maxIters = 1e3;
    opts2.maxit = 1e3;
    opts2.maxits = maxIters;
    opts2.restart = 0;
    opts2.doLM = 1;  % no LM
    opts2.doQN = 1;  opts2.mm  = 10;
    opts2.record = 1;

    %opts2.tau        = 4.;
    opts2.tau_adapt  = true;
    opts2.overwF_old = true;
    opts2.eps        = 0.85;
    opts2.x0 = x0;
    opts2.maxit      = 500;

    opts2.crit       = 1;
    opts2.tol        = 1e-13;
    [xopt,outopt]    = ssmNewtonL1Pen(A,At,b,N,mu,opts2);
    fopt = mu*norm(xopt,1)+0.5*norm(A(xopt)-b)^2;
    %fopt = mu*outopt.f_l1+outopt.f_Axmb;
    
    return
    
    opts1.fopt = fopt;
    opts2.fopt = fopt;
    opts3.fopt = fopt;

    for i = 1:length(tol) 
        opts2.crit       = 1;
        opts2.tol        = tol(i);
        [xtol,outtol] = ssmNewtonL1Pen(A,At,b,N,mu,opts2);
       % fsnf = mu*norm(xtol,1)+0.5*norm(A(xtol)-b)^2;
        fsnf = outtol.f;
        tolsnf = abs(fsnf - fopt)/max(1,abs(fopt))*1.1;
        tolsnf = max(tolsnf,1e-16);
        
         %% test SSM_Newton 
        opts2.x0 = x0;
        opts2.crit       = 2;
        opts2.tol        = tolsnf;
        opts2.maxit      = 300;

        [x3,out3]     = ssmNewtonL1Pen(A,At,b,N,mu,opts2);
        Acount(3,i) = Acount(3,i) + out3.Acalls;
        time(3,i) = time(3,i) + out3.time;
        
        opts2.maxit      = 500;
        [x4,out4]     = ssmNewtonL1Pen_proj(A,At,b,N,mu,opts2);
        Acount(4,i) = Acount(4,i) + out4.Acalls;
        time(4,i) = time(4,i) + out4.time;
        
        opts2.maxit      = 1000;
        opts2.mm = 2;
        [x7,out7]     = ssmNewtonL1Pen_LBFGSH(A,At,b,N,mu,opts2);
        Acount(7,i) = Acount(7,i) + out7.Acalls;
        time(7,i) = time(7,i) + out7.time;
        
        opts2.maxit      = 1000;
        opts2.mm = 1;
        [x8,out8]     = ssmNewtonL1Pen_LBFGSH(A,At,b,N,mu,opts2);
        Acount(8,i) = Acount(8,i) + out8.Acalls;
        time(8,i) = time(8,i) + out8.time;
            

        %% test SNF
        opts1.tau        = 4.;
        opts1.tau_adapt  = true;
        opts1.overwF_old = true;
        opts1.eps        = 0.85;
        opts1.crit       = 4;
        opts1.CG_adapt       = 1;
        
        opts1.x0 = x0;
        opts1.maxit      = 500;
        
        
         % run SNF
        opts1.tol        = tolsnf;
        opts1.crit       = 4;
        opts1.verbose    = 1;
        opts1.CG_adapt   = 0;
        [x1,out1]     = SNF(A,At,b,N,mu,opts1);
        Acount(1,i) = Acount(1,i) + out1.Acalls;
        time(1,i) = time(1,i) + out1.time;
        
         % run SNF
        opts1.tol        = tolsnf;
        opts1.crit       = 4;
        opts1.verbose    = 1;
        opts1.CG_adapt   = 1;
        [x2,out2]     = SNF(A,At,b,N,mu,opts1);
        Acount(2,i) = Acount(2,i) + out2.Acalls;
        time(2,i) = time(2,i) + out2.time;


        %% test FPC_AS
        Aop = A_operator(@(x) A(x), @(x) At(x));
        %opts3.gtol = tolsnf;
        opts3.testtol = tolsnf;
        opts3.testopt = 1;
        opts3.record = 1;
        opts3.mxitr = 1000;
        opts3.x0 = x0;

        [x5, out5] = FPC_AS(N,Aop,b,mu,[],opts3);
        Acount(5,i) = Acount(5,i) + out5.nfe + out5.nge + out5.nfe_sub + out5.nfe_sub;
        time(5,i) = time(5,i) + out5.cpu;



       %% test SpaRSA
        tolA = tolsnf;
        Psi = @(x,th) soft(x,th);   % denoising function
        Phi = @(x)    sum(abs(x(:)));     % regularizer

        [x6,out6]= SpaRSA(b,A,mu,...
            'Debias',0,...
            'AT', At, ... 
            'Phi',Phi,...
            'Psi',Psi,...
            'TRUE_F',fopt,...
            'MAXITERA',1000,...
            'Initialization',x0,...
            'StopCriterion',6,...
            'ToleranceA',tolA,...
            'Verbose', 1,...
            'Continuation',1,...
            'ContinuationSteps',40);
         %g4 = gradx(x4); res3 = resfun(x4,g4)

        Acount(6,i) = Acount(6,i) + out6.Acalls;
        time(6,i) = time(6,i) + out6.times(end);
         
    end
end

time = time/length(data.seed);
Acount = Acount/length(data.seed);
    



% 
% filename = strcat('./data/results_',num2str(dyna),'_tt',num2str(test_times));
% save(filename,'Acount','time','data','dyna')
