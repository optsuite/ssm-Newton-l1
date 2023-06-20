%clear all

 
 %addpath(genpath('../FPC_AS'));
 
 addpath(genpath('../../ssm-l1'));


dyna = 40;
%tol     = [1e+0,1e-2,1e-4];
tol = 1e-6;

%noise level
sigma = 0; 

    
% signal size
N       = 512^2;
% number of measurements
M       = floor(N/8);
% number of nonzeros
K       = floor(M/5);

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

nme = 4;
time = zeros(nme,length(tol));
Acount = zeros(nme,length(tol));


opts1 = [];
opts2 = [];
opts3 = [];
for i = 1:length(tol)
    counter1         = 0;
    counter2         = 0;
    counter3         = 0;
    time1 = 0;
    time2 = 0;
    time3 = 0;
    
    j = 1;
    
    mu = data.mu(j);
    seed = data.seed(j);
   % seed = 3262; %j=2
    [xs,b,~,R,Rt]    = createSignal(N,M,K,dyna,sigma,U,seed);
    
    A.times  = @(y) R(U(y));
    A.trans  = @(y) Ut(Rt(y));
    
    delta = 0;
    
    % optimality measure
    gradx  = @(x) x-At(A(x)-b);
    resfun = @(x,xgrad) norm(x - sign(xgrad).*max(abs(xgrad)-mu,0),'fro');
    objfun = @(x) 0.5*norm(A(x)-b,'fro')^2 + mu * norm(x,1);
    
    x0 = zeros(N,1);

    % call admm
    opts2 = [];
    opts2.record = 1;
    opts2.delta = delta;
    opts2.maxits =5000;
    opts2.tol = tol;
    opts2.record = 1;
    opts2.mu = 1;
    opts2.mu_update_itr = 15;

%     tic; [x1,Out1] = drs_l1(A, b, opts2); time1=toc
%     fprintf('||Ax-b|| = %6.3e: ||x||_1 = %6.3e\n',...
%         norm(A.times(x1)-b), norm(x1,1));
%     rerr = norm(x1-xs)/norm(xs);
%     fprintf('[nA,nAt]=[%i,%i]: Rel_err = %6.2e\n\n',...
%         Out1.cntA,Out1.cntAt,rerr)
%     counter1 = Out1.cntA + Out1.cntAt;

    
    % call ssmNewton
    opts2.doQN = 0; 
    opts2.maxit = 1000;
    opts2.maxits =1;
    opts2.tol = tol;
    opts2.maxNLS = 5;
    opts2.record = 1;

    tic; [x2,Out2] = ssmNewtonL1BPDb_adapt2(A, b, opts2); time2=toc
    fprintf('||Ax-b|| = %6.3e: ||x||_1 = %6.3e\n',...
        norm(A.times(x2)-b), norm(x2,1));
    rerr = norm(x2-xs)/norm(xs);
    fprintf('[nA,nAt]=[%i,%i]: Rel_err = %6.2e\n\n',...
        Out2.cntA,Out2.cntAt,rerr)
    counter2 = Out2.cntA + Out2.cntAt;
    
    
    
%     % call SPGL1
%     disp('--- SPGL1 ---');
%     spgl1A = createspgl1A(A.times,A.trans);
%     tolA = tol;
%     if ~exist('spgSetParms','file'); error('Solver SPGL1 is not found.'); end
%     spg_opts = spgSetParms('verbosity',1,'optTol',tolA);
%     tic; [x3,r,g,info] = spgl1(spgl1A,b,0,delta,x0,spg_opts); time3=toc
%     fprintf('||Ax-b|| = %6.3e: ||x||_1 = %6.3e\n',...
%         norm(A.times(x3)-b), norm(x3,1));
%     rerr3 = norm(x3-xs)/norm(xs);
%     fprintf('[nA,nAt]=[%i,%i]:  Rel_err = %6.2e\n\n',...
%         info.nProdA,info.nProdAt,rerr3);
%     counter3 = info.nProdA+info.nProdAt;     

    
    time(1,i) = time1;
    Acount(1,i) = counter1;
    time(2,i) = time2;
    Acount(2,i) = counter2;
    time(3,i) = time3;
    Acount(3,i) = counter3;

end


