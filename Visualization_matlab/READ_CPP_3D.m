clear
close all
clc

delete(gcp('nocreate'));

%% PRE-(POST-PROCESSING)
Show_AMR = true;
Show_Confidence_Region = true;
Show_Animation = false;
Save_Animation = false; % STILL NOT WORKING...

Name_var1 = 'S';
Name_var2 = 'I';
Name_var3 = 'R';

DIMENSIONS=3;
conf_lvl = 0.95;

%% CHANGE FOR YOUR CURRENT COMPUTER

Info=readcell  ('../SIMULATION_OUTPUT/Simulation_Info_5.csv');
Data=readmatrix('../SIMULATION_OUTPUT/Mean_PDFs_5.csv');

%%
Total_Pts           = Info{1,1};
Pts_Per_Dimension   = Info{1,2};

h_X =(Info{1,4}-Info{1,3})/(Pts_Per_Dimension-1);
X   =Info{1,3}:h_X:Info{1,4};

h_Y =(Info{1,6}-Info{1,5})/(Pts_Per_Dimension-1); 
Y   =Info{1,5}:h_X:Info{1,6};

h_Z =(Info{1,8}-Info{1,7})/(Pts_Per_Dimension-1); 
Z   =Info{1,7}:h_X:Info{1,8};

F_Output=zeros(Pts_Per_Dimension, Pts_Per_Dimension, Pts_Per_Dimension);

%%
[x,y,z] = meshgrid(X,Y,Z);

timesteps = length(Data)/Total_Pts;
skip = 3;
Integral_vals = zeros(1,floor(timesteps/skip));
iso_val       = zeros(floor(timesteps/skip),2);
Marg_X        = zeros(Pts_Per_Dimension,1);
Marg_Y        = Marg_X;
Marg_Z        = Marg_X;

for l = 1:skip:timesteps
    fprintf(['At step: ',num2str(l),newline]);
    parfor k=1:Pts_Per_Dimension
        for j=1:Pts_Per_Dimension
            for i=1:Pts_Per_Dimension
                i_aux=i+(j-1)*Pts_Per_Dimension+(k-1)*Pts_Per_Dimension^2+(l-1)*Total_Pts;
                F_Output(j,i,k) = Data(1,i_aux);
            end
        end
    end

    % COMPUTE MARGINAL DENSITIES
    Marg_X = h_Y*h_Z*reshape(sum(F_Output,[1 3]),[Pts_Per_Dimension,1]);
    Marg_Y = h_X*h_Z*reshape(sum(F_Output,[2 3]),[Pts_Per_Dimension,1]);
    Marg_Z = h_X*h_Y*reshape(sum(F_Output,[1 2]),[Pts_Per_Dimension,1]);

    Integral_vals(l)=sum(F_Output,'all')*h_X*h_Y*h_Z;

    % COMPUTE THE "HEIGHT" OF THE PDF OVER WHOM WE HAVE "conf_lvl" MASS
    iso_val(l,:)=ComputeRegion(Data(1,Pts_Per_Dimension^DIMENSIONS*(l-1)+1:Pts_Per_Dimension^DIMENSIONS*l),conf_lvl*Integral_vals(l),h_X,DIMENSIONS);

    figure(l)
    xlim([0,1]);xlabel(Name_var1);
    ylim([0,1]);ylabel(Name_var2);
    zlim([0,1]);zlabel(Name_var3);
    isosurface(x,y,z,F_Output,iso_val(l,1));view(3);colorbar;grid on; grid minor; lightangle(-45,45); drawnow;

    figure(100+l)
    subplot(3,1,1)
    plot(X,Marg_X,'r.-');xlabel(Name_var1);ylabel('Probability Density'); drawnow;
    subplot(3,1,2)
    plot(Y,Marg_Y,'r.-');xlabel(Name_var2);ylabel('Probability Density'); drawnow;
    subplot(3,1,3)
    plot(Z,Marg_Z,'r.-');xlabel(Name_var3);ylabel('Probability Density'); drawnow;
    pause(1);
end
