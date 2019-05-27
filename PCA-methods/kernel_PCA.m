function [Xpc,pc,expvar]=kernel_PCA(f,h)

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

%% Build Kernel distance matrix across time dimension
k=norm2mat(x',x');

%% Sigma definition; sigma ~ median distance
sigma=median(abs(k(:)));                    % mean distance

%% Kernel PCA (RBF matrix)
K=exp(-k/(2*sigma.^2));                     % kernel function
K=H*K*H;                                    % Centering Kernel matrix
[pc,expvar]=eig(K);
[expvar,idx]=sort(real(diag(expvar)),'descend');
pc=pc(:,idx);                               % Sort Pcs
pc=pc(:,1:h);
expvar=expvar(1:h);

%% spatial projection (Covariance approach)
Xpc=pc'*x;
Xpc=reshape(Xpc,h,nx,ny);
end

%% Compute the distance matrix
function D = norm2mat(X1,X2)
    D = - 2 * (X1' * X2);
    D = bsxfun(@plus, D, sum(X1.^2,1)');
    D = bsxfun(@plus, D, sum(X2.^2,1));
end