function []=plot_features(Xpc,pc,expv,Title)

h=size(expv,1);
expv=expv*100/sum(expv);

%% Spatial and Temporal modes plot
map=cmocean('balance');
figure,
for i=1:h
    subplot(h,2,2*i),
    plot(real(pc(:,i)),'-r','linewidth',.5);
    title(['Time Feature $\#',num2str(i),'(',num2str(expv(i)),'\%)$'],'interpreter','latex');
    grid;

    ax1=subplot(h,2,2*i-1);
    xr=real(squeeze(Xpc(i,:,:)));
    contourf(xr);
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    grid;title(['Spatial Feature $\#',num2str(i),'(',num2str(expv(i)),'\%)$'],'interpreter','latex');
    colormap(ax1,map);colorbar;
    axis square;
    
end
suptitle(Title);
end