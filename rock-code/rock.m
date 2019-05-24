%% ROCK-PCA (ROtated Kernel Complex PCA)
%%% for Spatio-temporal data analysis
%%% Diego Bueso, IPL, Universitat de Valencia
%%% diego.bueso@uv.es || (22/05/2019)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Xpc,pc,expvar,Sigma,Kernel]=rock(f,h,Pk,N)

%% IN PUT
%f --> Spatio temporal data f(Temporal, x Spatial, y Spatial)
%h --> number of PC's [2,3,...,h] 
%N --> number of sigmas to try 
%Pk -> number of powers to try [1,2,...,Pk]


%% OUT PUT
%Xpc    --> Projected ROCK pc
%pc     --> optimus pricipal component from ROCK
%expvar --> Explained variances
%Sigma  --> optimus sigma
%Kernel --> estimated kernel matrix

%% Hilbert transform (complex approach)
[nt,nx,ny]=size(f);
H=eye(nt)-(1/nt)*ones(nt);     % Centering matrix

x=reshape(f,nt,nx*ny);
x=hilbert(x);
x=H*x;                         % Centering data

%% Build Kernel distance matrix across time dimension
k=norm2mat(x',x');
disp('Building kernel matrix');
disp('100 %');

%% Sigma definition; Boundary: median/20 --> maximun·10
m=median(abs(k(:)));    % mean distance
M=max(abs(k(:)));       % maximun distance
sigma=linspace(m/10,M*10,N);

%% Kernel matrix RBF contruction
K=zeros(nt,nt,N);                           % Kernel matrix for each sigma
Pc=zeros(nt,h,N);                           % PC for each sigma
for i=1:N
  K(:,:,i)=exp(-k/(2*sigma(i).^2));         % kernel function              
  K(:,:,i)=H*K(:,:,i)*H;                    % Centering Kernel matrix  
  [v,l]=eig(squeeze(K(:,:,i)));             
  [~,idx]=sort(diag(l),'descend');          
  v=v(:,idx);                               % Sort Pcs                      
  v=v(:,1:h);
  Pc(:,:,i)=v;
  clc;
  disp('Building kernel matrix');
  disp('100 %');
  disp('solving Complex RBF');
  disp([num2str(i*100/N),' %']);
end

%% Kurtosis optimize criterion (number of Components and Sigma)
P=Pk+1;                     % num of power to try
hh=2:h;                     % num of PC to try
Kurtosis=zeros(h-1,N);

for i=hh
for j=1:N
Pcr=Pc(:,1:i,j);
for kk=1:i
     Pcr(:,kk)=Pcr(:,kk)/(Pcr(:,kk)'*Pcr(:,kk));  % Normalize pc's
end
Kurtosis(i,j)=(nt/i)*sum(sum(real(Pcr).^4)./(sum(real(Pcr.^2).^2))); % kurtosis
end
clc;
disp('Building kernel matrix');
disp('100 %');
disp('solving Complex RBF');
disp('100 %');
disp('Kurtosis Optimization');
disp([num2str(i*100/(h+P)),' %']);
end

[~,Npc]=max(max(Kurtosis,[],2));                                % find the number of PC's
[~,Sigma]=max(max(Kurtosis));                                   % find the Sigma

%% Kurtosis optimize criterion (Power of promax rotation)
pc=squeeze(Pc(:,1:Npc,Sigma));
Kurtosis=zeros(P,1);
for p=1:P   
power=p-1;
if power==0
    Pcr=pc;
else
    Pcr=fft(pc);                           % promax rotation on the frequency space 
    Rv=Promax(Pcr,power);
    Rv=(abs(Rv)./abs(Pcr)).*Pcr;
    Rv=hilbert(real(ifft(Rv)));
    Pcr=H*Rv;
end
for kk=1:Npc
     Pcr(:,kk)=Pcr(:,kk)/(Pcr(:,kk)'*Pcr(:,kk));
end
Kurtosis(p)=(nt/i)*sum(sum(real(Pcr).^4)./(sum(real(Pcr.^2).^2)));
clc;
disp('Building kernel matrix');
disp('100 %');
disp('solving Complex RBF');
disp('100 %');
disp('Kurtosis Optimization');
disp([num2str((h+p)*100/(h+P)),' %']);
end

[~,Power]=max(Kurtosis);                        % find the power of promax rotation

disp(['Power: ',num2str(Power)]);
disp(['PCs: ',num2str(Npc)]);
disp(['median(dk)/Sigma: ',num2str(m/sigma(Sigma))]);
disp(['max(dk)/Sigma: ',num2str(M/sigma(Sigma))]);
disp(['sigma: ',num2str(Sigma)]);

%% extract the rotated component
if Power==0
    pc=squeeze(Pc(:,1:Npc,Sigma));
else
    Pcr=squeeze(Pc(:,1:Npc,Sigma));
    Pcr=fft(Pcr);
    Rv=Promax(Pcr,Power);
    Rv=(abs(Rv)./abs(Pcr)).*Pcr;
    Rv=hilbert(real(ifft(Rv)));
    pc=H*Rv;
end

%% reconstruction eigen values
expvar=zeros(Npc,1);
for i=1:Npc
    expvar(i)=norm(pc(:,i)'*K(:,:,Sigma))/norm(pc(:,i));
end
[expvar,idx]=sort(expvar,'descend');
pc=pc(:,idx);

%% spatial projection (covariance approach)
Xpc=pc'*x;
Xpc=reshape(Xpc,Npc,nx,ny);

%% return kernel matrix and his sigma parameter
Kernel=squeeze(K(:,:,Sigma));
Sigma=sigma(Sigma);

end

%% Compute the distance matrix
function D = norm2mat(X1,X2)
    D = - 2 * (X1' * X2);
    D = bsxfun(@plus, D, sum(X1.^2,1)');
    D = bsxfun(@plus, D, sum(X2.^2,1));
end