
function [Xpc,pc,expvar]=SSA(f,h,M)

%% IN PUT
%f --> Spatio temporal data f(Temporal, x Spatial, y Spatial)
%h --> number of PC's [2,3,...,h] 
%M --> length of time window (delay)

%% OUT PUT
%Xpc    --> Projected pc
%pc     --> principal components
%expvar --> Explained variances

%% Preprocess
[nt,nx,ny]=size(f);
H=eye(nt)-(1/nt)*ones(nt);     % Centering matrix
x=reshape(f,nt,nx*ny);
x=H*x;

%% Size
[N,D]=size(x);

%% M-SSA analysis
% Perform M-SSA with the Broomhead-King approach of a covariance-matrix% estimation |C|. 

idx=hankel(1:N-M+1,N-M+1:N); % index matrix for time-delayed embedding
xtde=zeros(N-M+1,M,D);       % embedding matrix of size [N-M+1,M] for D channels
for d=1:D
  xnew=x(:,d);
  xtde(:,:,d)=xnew(idx);
end
xtde=reshape(xtde,N-M+1,D*M,1); % reshape to form full trajectory matrix

% M-SSA analysis
C=xtde'*xtde/(N-M+1);           % Broomhead and King (1986)
[pc,expvar]=eigs(C,h);
expvar=diag(expvar);
[expvar,idx]=sort(expvar,'descend');
pc=pc(:,idx);
expvar=expvar*100/sum(expvar);
pc=xtde*pc;

% spatial projection
Xpc=pc'*x(1:N-M+1,:);
Xpc=reshape(Xpc,h,nx,ny);
end
