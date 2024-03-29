clear
close all
clc

delete(gcp('nocreate'));

%% PRE-(POST-PROCESSING)
Show_AMR = true;
Show_Confidence_Region = true;
Show_Animation = false;
Save_Animation = false; % STILL NOT WORKING...I'LL FIND OUT SOON

Name_var1 = 'A_P';
Name_var2 = 'A_E';

%% CHANGE FOR YOUR CURRENT COMPUTER

Info=readcell('Simulation_Info.csv');
Data=readmatrix('Mean_PDFs.csv');

Points_X = [0,1,2,3,4,5,6,7];
Points_Y = [0.018,0.941,0.868,0.787,0.759,0.39,0.4,0.246];

%% READ C++ OUTPUT
Total_Pts           = Info{1,1};
Pts_Per_Dimension   = Info{1,2};

h_X =(Info{1,4}-Info{1,3})/(Pts_Per_Dimension-1);
X   =Info{1,3}:h_X:Info{1,4};

h_Y =(Info{1,6}-Info{1,5})/(Pts_Per_Dimension-1); 
Y   =Info{1,5}:h_X:Info{1,6};

F_Output=zeros(Pts_Per_Dimension, Pts_Per_Dimension);

aux = size(Info)-1;
t=zeros(aux(2),1);

Integral_values=zeros(aux(2),1);

parpool('threads');
Marg=zeros(Pts_Per_Dimension+1,length(t));

Stats_2D    = zeros(length(t),5);
Stats_1D_X  = zeros(length(t),6);
Stats_1D_Y  = zeros(length(t),6);

for k=1:aux(2)

    t(k)=Info{2,k};
    parfor j=1:Pts_Per_Dimension
        for i=1:Pts_Per_Dimension
            i_aux=i+(j-1)*Pts_Per_Dimension+(k-1)*Total_Pts;
            F_Output(i,j) = Data(1,i_aux);
        end
    end
    
    Integral_values(k)=h_X*h_Y*sum(F_Output(:,:),'all');

    f=figure(1); % show the interpolated function and the confidence region obtained with the bisection method
    
    % MARGINAL DENISTY
    for i=1:Pts_Per_Dimension
        MargX(i,k) = sum(F_Output(i,:)).*h_X;
    end
    Stats_1D_X(k,:) = Stats(MargX(:,k),X,0.95);
    for i=1:Pts_Per_Dimension
        MargY(i,k)  = sum(F_Output(:,i)).*h_Y;
    end
    Stats_1D_Y(k,:) = Stats(MargY(:,k),Y,0.95);

    % DO WE WANT AMR GRAPH?
    if Show_AMR
        % SHOW AMR IN EACH CASE
        [MeshId,val,GridPt,f_Disc] = AMR(F_Output(:,:),log2(Pts_Per_Dimension),0,X,Y,3.5e-4);

        % Function with the confidence region curve
        subplot(1,2,1)
%         contour(X,Y,F_Output(:,:),25);view(0,90); Mesh on; Mesh minor;
        mesh(X,Y,F_Output(:,:));view(0,90); Mesh on; Mesh minor;colormap('jet')
        title(['Current time: ',num2str(t(k))]); colorbar;
        ylabel(Name_var1);xlabel(Name_var2);
        
        if Show_Confidence_Region
            hold on;
       
            % Compute the confidence region for each time step shown in graphics %%
            confidenceLvl = 0.95;
            confidenceLvl = confidenceLvl * (Integral_values(k));
        
            f_low = ComputeRegion(F_Output(:,:),confidenceLvl,X(end)-X(1),Y(end)-Y(1),h_X); % 1 for the ensemble region
            [~,c]=contour(X,Y,F_Output(:,:),[f_low,f_low],'r','ShowText','off');
            c.LineWidth=2;
            c.ZLocation = f_low;
            
            hold off;
        end
    
        % Adaptive Mesh Refinement output
        subplot(1,2,2)
        M=sparse(val); % Perfect!!!!
        spy(M);view(0,-90);
        
        f.Position(3:4) = [1000,400]; % the correct form is 5:2 ( = 10:4 ...with 1.2 scaling in this case)
    else
        mesh(X,Y,F_Output(:,:));view(0,90); Mesh on; Mesh minor;colormap('jet')
        title(['Current time: ',num2str(t(k))]); colorbar;
        ylabel(Name_var1);xlabel(Name_var2);
        
        if Show_Confidence_Region
            hold on;
       
            % Compute the confidence region for each time step shown in graphics %%
            confidenceLvl = 0.95;
            confidenceLvl = confidenceLvl * (Integral_values(k));
        
            f_low = ComputeRegion(F_Output(:,:),confidenceLvl,X(end)-X(1),Y(end)-Y(1),h_X); % 1 for the ensemble region
            [~,c]=contour(X,Y,F_Output(:,:),[f_low,f_low],'r','ShowText','off');
            c.LineWidth=2;
            c.ZLocation = f_low;
            
            hold off;
        end
    end
    drawnow;
    
    if Show_Animation
    % to prepare the animation to be repeated afterwards 
        ax = gca;
        ax.Units = 'pixels';
        pos = ax.Position;
        ti = ax.TightInset;
    
        rect = [-ti(1), -ti(2), pos(3)+ti(1)+ti(3), pos(4)+ti(2)+ti(4)];
        Mov(k) = getframe(ax,rect);
    end
    
    % stats
    Stats_2D(k,:)=StatInfo(F_Output(:,:),X,Y,h_X);
%     std_X(k)=sum((X(:)-Stats_2D(k,2)).^2.*MargX(:,k))*h_X;
%     std_X(k)=sqrt(std_X(k));
% 
%     std_Y(k)=sum((Y(:)-Stats_2D(k,3)).^2.*MargY(:,k))*h_Y;
%     std_Y(k)=sqrt(std_Y(k));   
end

%% FIGURES
if Show_Animation
    figure(5)
    movie(figure(5),Mov,1,10)
end

if Save_Animation
    writeAnimation('loop.gif','FrameRate',10,'LoopCount',3);
end

% figure(2)
% % plot(t(:),Stats_1D_X(:,1),'.-',t(:),ones(length(t),1).*mean(Stats_1D_X(:,1)),'-');
% % xlabel('Time');ylabel('Total mass');legend('Integral value','Mean Int. Value')
% 
% plot(t(:),Stats_1D_X(:,2),'.-',t(:), Stats_1D_Y(:,2),'.-');
% xlabel('Time');ylabel('');legend(Name_var1,Name_var2);

f=figure(3);

f.Position(3:4) = [575,720];
subplot(2,1,1)
plot(t(:),Stats_1D_X(:,2),'r.-',...
    t(:),Stats_1D_X(:,5),'k--',...
    t(:),Stats_1D_X(:,6),'k--');
xlabel('Time');ylabel(Name_var1);
legend('Mean','Lower CI','Upper CI');
subplot(2,1,2)
hold on;
plot(t(:),Stats_1D_Y(:,2),'r.-',...
    t(:),Stats_1D_Y(:,5),'k--',...
    t(:),Stats_1D_Y(:,6),'k--');
xlabel('Time');ylabel(Name_var2);
legend('Mean','Lower CI','Upper CI');

% plot(Points_X,Points_Y,'k.','markersize', 15,'DisplayName','Data');

hold off;


figure(4)
plot(t(:),Stats_2D(:,4),'.-',t(:),Stats_2D(:,5),'-');xlabel('Time');legend('Auto-variance','Covariance')

%% Graphs 2
figure(2)
time = t(1:20:end);

temp_Y = zeros(length(Y),length(time));

temp_Y(:,:)= MargY(:,1:20:end);
[X_mesh,T_mesh]=meshgrid(Y,time);
w=waterfall(X_mesh,T_mesh,temp_Y(:,:)');
w.EdgeColor = 'b';
w.EdgeAlpha = 0.4;
w.FaceColor = 'b';
w.FaceAlpha = 0.2;
xlabel(Name_var2)
ylabel('Time (t)')
zlabel('Probability Density')
xlim([0,0.1])
ylim([0,time(end)])
Mesh off
hold on;

p1=plot3(Stats_1D_Y(:,2),t,zeros(1,length(t)),'r.-');
p2=plot3(Stats_1D_Y(:,5),t,zeros(1,length(t)),'k-.');
p3=plot3(Stats_1D_Y(:,6),t,zeros(1,length(t)),'k-.');

p1.LineWidth=1;
p2.LineWidth=1;
p3.LineWidth=1;
view([52,-72]);
hold off;

% save('2D_VanDerPol_Oscillator.mat')