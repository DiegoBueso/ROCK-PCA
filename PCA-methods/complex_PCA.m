function [Xpc,pc,expvar]=complex_PCA(f,h)

%% IN PUT
%f --> Spatio temporal data f(Temporal, x Spatial, y Spatial)
%h --> number of PC's [2,3,...,h] 

%% OUT PUT
%Xpc    --> Projected pc
%pc     --> principal components
%expvar --> Explained variances

%% Preprocess
[nt,nx,ny]=size(f);
H=eye(nt)-(1/nt)*ones(nt);     % Centering matrix
x=reshape(f,nt,nx*ny);
x=hilbert(x);

%% PCA
Gram=x*x';          
Gram=H*Gram*H;                                   % Centering Gram matrix  
[pc,expvar]=eig(Gram);             
[expvar,idx]=sort(real(diag(expvar)),'descend');          
pc=pc(:,idx);                                    % Sort Pcs                      
pc=pc(:,1:h);
expvar=expvar(1:h);

%% spatial projection
Xpc=pc'*x;
Xpc=reshape(Xpc,h,nx,ny);
end