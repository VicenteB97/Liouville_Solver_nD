impulse_times=[7,14,27];

close all
clc

for k=1:length(impulse_times)

    f=figure(300+k);
    f.Position(3:4)=[1200,480];
    idx = impulse_times(k);

    subplot(1,2,1)
        contour(X,Y,F_Output(:,:,idx),25);title(['Before impulse. Time: ',num2str(t(idx))]);
        view(0,90);colorbar;ylabel('Position');xlabel('Velocity');hold on;
        [~,c]=contour(X,Y,F_Output(:,:,idx),[f_low,f_low],'r','ShowText','off');
        c.LineWidth=1;hold off;

    subplot(1,2,2)
        contour(X,Y,F_Output(:,:,idx+1),25);title(['After impulse. Time: ',num2str(t(idx))]);
        view(0,90);colorbar;ylabel('Position');xlabel('Velocity');hold on;
        [~,c]=contour(X,Y,F_Output(:,:,idx+1),[f_low,f_low],'r','ShowText','off');
        c.LineWidth=1;hold off;
end