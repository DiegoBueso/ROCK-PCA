clc; close all; clear;
addpath([cd,'/../rock-code']);
%% demo
n=18;   % Spatial samples
nt=300; % Time samples

%% AR time series
load AR

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
      c=AR(k);
      X(k,i,j)=(a*0.5+b*0.5)*cos(0.5*rr)+b*cos(0.5*xx*yy)+c*cos(yy);
    end
  end
end

%% prepare data and ROCK!
h=4;        % maximun number of features to search
P=10;       % maximun power for the promax 
N=100;      % number of sigmas to search (from 0.1 of the mean distance and 10 of the maximun distance
[Xpc,pc,expvar,Sigma,Kernel]=rock(X,h,P,N);

%% contruct the theoretical data
ft=zeros(nt,3);
ft(:,1)=sin(16*t);
ft(:,2)=AR;
ft(:,3)=exp(-abs(t));

fs=zeros(3,n,n);
for i=1:n
    for j=1:n
        xx=x(i);yy=y(j);tt=t(k);
        rr=sqrt((xx-0.5)^2+(yy-0.5)^2);
        fs(1,i,j)=cos(0.5*xx*yy);
        fs(2,i,j)=cos(yy);
        fs(3,i,j)=cos(0.5*rr);
    end
end

aux=reshape(X,nt,n*n);
H=eye(nt)-ones(nt)/nt;
aux=H*aux;
expf=zeros(3,1);
for i=1:3
    expf(i)=norm(ft(:,i)'*(aux*aux'))./norm(ft(:,i));
end

%% plot results
plot_rock_features(fs,ft,expf)        % Theoretical features
plot_rock_features(Xpc,pc,expvar)     % Rock-pca features
tile;
