function [Xpc,pc,expvar]=promax_PCA(f,h,power)

%% IN PUT
%f --> Spatio temporal data f(Temporal, x Spatial, y Spatial)
%h --> number of PC's  [2,3,...,h]
%power -> promax power

%% OUT PUT
%Xpc    --> Projected pc
%pc     --> principal components
%expvar --> Explained variances

%% Preprocess
[nt,nx,ny]=size(f);
H=eye(nt)-(1/nt)*ones(nt);     % Centering matrix
x=reshape(f,nt,nx*ny);

%% PCA
Gram=x*x';          
Gram=H*Gram*H;                              % Centering Gram matrix  
[pc,expvar]=eig(Gram);             
[~,idx]=sort(real(diag(expvar)),'descend');          
pc=pc(:,idx);                               % Sort Pcs                      
pc=real(pc(:,1:h));

%% promax ( vzarimax -> power=1)
pc2=pc;
pc2=fft(pc2);
Rpc=Promax(pc2,power);
Rpc=(abs(Rpc)./abs(pc2)).*pc2;
Rpc=real(ifft(Rpc));
pc=H*Rpc;

%% reconstruction eigen values
expvar=zeros(h,1);
for i=1:h
    expvar(i)=norm(pc(:,i)'*Gram)/norm(pc(:,i));
end
[expvar,idx]=sort(expvar,'descend');
pc=pc(:,idx);

%% spatial projection (covariance approach)
Xpc=pc'*x;
Xpc=reshape(Xpc,h,nx,ny);

end