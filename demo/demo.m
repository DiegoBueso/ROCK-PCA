clc; close all; clear;
addpath([cd,'/../rock-code']);
addpath([cd,'/../PCA-methods']);

%% demo
n=18;   % Spatial samples
nt=300; % Time samples

%% load theoretical data (from spatio-temporal model)
load theoretical_data
AR=ft(:,2);

%% spatio-temporal model (nt x n x n)
t=linspace(-pi*2,pi*2,nt);
x=linspace(0,1,n);
y=linspace(0,1,n);
X=zeros(nt,n,n);
for i=1:n
  for j=1:n
    for k=1:nt
      xx=x(i);yy=y(j);tt=t(k);
      rr=sqrt((xx-0.5)^2+(yy-0.5)^2);
      a=exp(-abs(tt));
      b=sin(16*tt);
      c=-AR(k);
      X(k,i,j)=(a*0.5+b*0.5)*cos(0.5*rr)+b*cos(0.5*xx*yy)+c*cos(yy);
    end
  end
end

%% METHODS
%% PCA
h=4;        % number of features
[XpcPCA,pcPCA,expvarPCA]=PCA(X,h);

%% complex PCA
h=4;        % number of features
[XpcCPCA,pcCPCA,expvarCPCA]=complex_PCA(X,h);

%% promax PCA (power=1 -> Varimax)
h=4;        % number of features
P=1;        % maximun power for the promax 
[XpcR,pcR,expvarR]=promax_PCA(X,h,P);

%% SSA
h=4;        % number of features
M=5;       % length of time window (delay)
[XpcS,pcS,expvarS]=SSA(X,h,M);

%% Kernel PCA (sigma chosen by mean distance)
h=4;        % number of features
[XpcK,pcK,expvarK]=kernel_PCA(X,h);

%% ROCK PCA!!! (set boundary parameters)
h=4;        % maximun number of features to search
P=10;       % maximun power for the promax 
N=100;      % number of sigmas to search (from 0.1 of the mean distance and 10 of the maximun distance)
[Xpc,pc,expvar,Sigma,Kernel]=rock(X,h,P,N);


%% RESULTS
%% plot results
plot_features(fs,ft,expf,'Original signals')              % Theoretical features
plot_features(XpcPCA,pcPCA,expvarPCA,'PCA')               % pca features
plot_features(XpcCPCA,pcCPCA,expvarCPCA,'Complex PCA')    % complex pca features
plot_features(XpcR,pcR,expvarR,'Promax PCA')              % varimax pca features
plot_features(XpcK,pcK,expvarK,'Kernel PCA')              % kernel pca features
plot_features(XpcS,pcS,expvarS,'SSA')                     % SSA features
plot_features(Xpc,pc,expvar,'ROCK PCA')                   % Rock-pca features
tile;
