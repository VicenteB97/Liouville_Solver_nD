clear
close all
clc

delete(gcp('nocreate'));

%% CHANGE FOR THIS CURRENT COMPUTER

Info=readcell('C:\Users\Imm-Vicente\source\repos\Liouville_Solver_nD\SOURCE_FILES\build\Simulation_Info.csv');
Data=readmatrix('C:\Users\Imm-Vicente\source\repos\Liouville_Solver_nD\SOURCE_FILES\build\Mean_PDFs.csv');

%% GATHER INFO
Total_Points = Info{1,1};

Time_values = Info{1,5};
% Time_values = 27;

time = zeros(Time_values,1);

X0=Info{1,3};
XF=Info{1,4};
X=X0:(XF-X0)/(Total_Points - 1):XF;

f_output = zeros(Total_Points,Time_values);
Integral_vals = zeros(Time_values,1);

figure(1)
hold on;
for i = 1:Time_values
    f_output(:,i) = Data((i-1) * Total_Points + 1:i*Total_Points);
    plot(X,f_output(:,i));drawnow;

    Integral_vals(i) = (X(2)-X(1))* sum(f_output(:,i),1);
    time(i)=Info{2,i};

end
hold off;

figure(2)
[X_mesh,T_mesh]=meshgrid(X,time);
w=waterfall(X_mesh,T_mesh,f_output');
w.EdgeColor = 'b';
w.EdgeAlpha = 0.3;
w.FaceColor = 'b';
w.FaceAlpha = 0.65;
xlabel('VarX (X)')
ylabel('Time (t)')
zlabel('Probability Density')
xlim([0,1.8])
ylim([0,time(end)])
hold on;


aux_time_counter=[];
stats_output=[];
for k=1:length(time)        
    stats_output = vertcat(stats_output,Stats(f_output(:,k),X,0.95));

    aux_time_counter=horzcat(aux_time_counter,time(k));
end

% 2.2.- Mean and confidence interval
aux_time_counter=sort(aux_time_counter);
p1=plot3(stats_output(:,2),aux_time_counter,zeros(1,length(aux_time_counter)),'r.-');
p2=plot3(stats_output(:,5),aux_time_counter,zeros(1,length(aux_time_counter)),'k-.');
p3=plot3(stats_output(:,6),aux_time_counter,zeros(1,length(aux_time_counter)),'k-.');

p1.LineWidth=1;
p2.LineWidth=1;
p3.LineWidth=1;
view([52,72]);
hold off;


figure(3)
hold on;
for k=1:length(aux_time_counter)
    plot(aux_time_counter,stats_output(:,2),'r.-');
    plot(aux_time_counter,stats_output(:,5),'k-.');
    plot(aux_time_counter,stats_output(:,6),'k-.');
end
legend('Mean','CI lower bound','CI upper bound','Location','northwest');xlabel('Time (t)'); ylabel('VarX (X)'); hold off;

figure(4)
hold on;
for k=1:length(aux_time_counter)
    plot(aux_time_counter,stats_output(:,6)-stats_output(:,5),'r.-');
    plot(aux_time_counter,stats_output(:,3),'k.-');
end
legend('CI amplitude','Std. Deviation');xlabel('Time (t)'); ylabel('VarX (X)'); hold off;

%%
figure(5)
subplot(3,1,1)
plot(X,f_output(:,16),X,f_output(:,17)); legend('f(x,T_1^-)','f(x,T_1)');
xlabel('X(t)');ylabel('Pr. density');xlim([0,3]);

subplot(3,1,2)
plot(X,f_output(:,27),X,f_output(:,28)); legend('f(x,T_2^-)','f(x,T_2)');
xlabel('X(t)');ylabel('Pr. density');xlim([0,3]);

subplot(3,1,3)
plot(X,f_output(:,38),X,f_output(:,39)); legend('f(x,T_3^-)','f(x,T_3)');
xlabel('X(t)');ylabel('Pr. density');xlim([0,3]);