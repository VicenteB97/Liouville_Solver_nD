clear
% close all
clc

delete(gcp('nocreate'));

%% CHANGE FOR THIS CURRENT COMPUTER

Info=readcell('C:\Users\Imm-Vicente\source\repos\CMAKE_Liouville_Solver\build\Simulation_Info.csv');
Data=readmatrix('C:\Users\Imm-Vicente\source\repos\CMAKE_Liouville_Solver\build\Mean_PDFs.csv');

% load('2D_Duffing.mat')

% Info=readcell('C:\Users\Vicentin\source\repos\CMAKE_LIOUVILLE\build\Simulation_Info.csv');
% Data=readmatrix('C:\Users\Vicentin\source\repos\CMAKE_LIOUVILLE\build\Mean_PDFs.csv');
%% READ C++ OUTPUT
Total_Pts           = Info{1,1};
Pts_Per_Dimension   = Info{1,2};

h_X =(Info{1,4}-Info{1,3})/(Pts_Per_Dimension-1);
X   =Info{1,3}:h_X:Info{1,4};

h_Y =(Info{1,6}-Info{1,5})/(Pts_Per_Dimension-1); 
Y   =Info{1,5}:h_X:Info{1,6};

F_Output=zeros(Pts_Per_Dimension,Pts_Per_Dimension,Info{1,7});
t=zeros(Info{1,7},1);

Integral_values=zeros(Info{1,7},1);

parpool('threads');

for k=1:Info{1,7}
    t(k)=Info{2,k};
    %F_Output=zeros(Pts_Per_Dimension,Pts_Per_Dimension);
    parfor j=1:Pts_Per_Dimension
        for i=1:Pts_Per_Dimension
            i_aux=i+(j-1)*Pts_Per_Dimension+(k-1)*Total_Pts;
            F_Output(i,j,k) = Data(1,i_aux);
        end
    end

%     [MeshId,val,GridPt,f0_Disc] = AMR(F_Output(:,:,k),9,1,X,Y,1e-6);
% 
%     figure(k)
%     contour(X,Y,F_Output(:,:,k),15);colorbar;
%     title(['Current time: ',num2str(t(k))]);drawnow;
    
    Integral_values(k)=h_X*h_Y*sum(F_Output(:,:,k),'all');
   
end
%% GRAPHS 1
Marg=zeros(Pts_Per_Dimension+1,length(t));

Stats=zeros(length(t),5);

h3=waitbar(0,'Post-processing: Computing graphical and statistical output...');
for k=1:length(t)
    waitbar(k/length(t),h3);
    
    % SHOW AMR IN EACH CASE
    [MeshId,val,GridPt,f_Disc] = AMR(F_Output(:,:,k),log2(Pts_Per_Dimension),0,X,Y,5*1e-4);

    f=figure(100);%(k+1); % show the interpolated function and the confidence region obtained with the bisection method
    
    % MARGINAL DENISTY
    for i=1:Pts_Per_Dimension
        MargX(i,k)=sum(F_Output(i,:,k)).*h_X;
    end
    for i=1:Pts_Per_Dimension
        MargY(i,k)=sum(F_Output(:,i,k)).*h_Y;
    end

    % Function with the confidence region curve
    subplot(1,2,1)
    contour(X,Y,F_Output(:,:,k),25);view(0,90); grid on; grid minor;
    title(['Current time: ',num2str(t(k))]);colorbar;ylabel('Position');xlabel('Velocity');
    hold on;
   
    % Compute the confidence region for each time step shown in graphics %%
    confidenceLvl = 0.95;
    confidenceLvl = confidenceLvl * (Integral_values(k));

    f_low = ComputeRegion(F_Output(:,:,k),confidenceLvl,X(end)-X(1),Y(end)-Y(1),h_X); % 1 for the ensemble region
    [~,c]=contour(X,Y,F_Output(:,:,k),[f_low,f_low],'r','ShowText','off');
    c.LineWidth=1;hold off;

    % Adaptive Mesh Refinement
    subplot(1,2,2)
    M=sparse(val); % Perfect!!!!
    spy(M);view(0,-90);

    f.Position(3:4) = [1000,400]; % the correct form is 5:2 ( = 10:4 ...with 1.2 scaling in this case)
    drawnow; pause(1)
    
    % stats
    Stats(k,:)=StatInfo(F_Output(:,:,k),X,Y,h_X);
    std_X(k)=sum((X(:)-Stats(k,2)).^2.*MargX(:,k))*h_X;
    std_X(k)=sqrt(std_X(k));

    std_Y(k)=sum((Y(:)-Stats(k,3)).^2.*MargY(:,k))*h_Y;
    std_Y(k)=sqrt(std_Y(k));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

close(h3);

%% GRAPHS 2

% MARGINAL PDFs
% k=min(length(t),5);

for k=1:length(t)
    figure(101)
    plot(X,MargX(:,k)); hold on;
    plot(Y,MargY(:,k)); hold off;
    pause(1)
end

Marginal_Variable=1; % 1 for X, and 2 for Y
Variable_Initial=0.4;
Variable_Final=0.6;
ShowPlot=1; % 1 true, 0 false

% Marg=MarginalPlots(F_Output(:,:,k),X,Y,MargX(:,k),MargY(:,k),Marginal_Variable,Variable_Initial,Variable_Final,1,length(t));

%%
figure(4+length(t))
%subplot(2,2,1)
plot(t(:),Stats(:,1),'.-',t(:),ones(length(t),1).*mean(Stats(:,1)),'-');
xlabel('Time');ylabel('Total mass');legend('Integral value','Mean Int. Value')

plot(t(:),Stats(:,2),'.-',t(:), Stats(:,3),'.-');
xlabel('Time');ylabel('');legend('Mean curve 1','Mean curve 2');

f=figure(5+length(t));

f.Position(3:4) = [575,720];
subplot(2,1,1)
plot(t(:),Stats(:,2),'r.-',t(:),Stats(:,2)+2.*std_X(:),'k--',...
    t(:),Stats(:,2)-2.*std_X(:),'k--');xlabel('Time');ylabel('Mean Var1');
legend('Mean','Mean + Std','Mean - Std');
subplot(2,1,2)
plot(t(:),Stats(:,3),'r.-',t(:),Stats(:,3)+2.*std_Y(:),'k--',...
    t(:),Stats(:,3)-2.*std_Y(:),'k--');xlabel('Time');ylabel('Mean Var2');
legend('Mean','Mean + Std','Mean - Std');


figure(3+length(t))
plot(t(:),Stats(:,4),'.-',t(:),Stats(:,5),'-');xlabel('Time');legend('Auto-variance','Covariance')