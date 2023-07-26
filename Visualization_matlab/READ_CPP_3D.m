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

Info=readcell  ('../SIMULATION_OUTPUT/Simulation_Info_0.csv');

Total_Pts           = Info{1,1};
Pts_Per_Dimension   = Info{1,2};

h_X =(Info{1,4}-Info{1,3})/(Pts_Per_Dimension-1);
X   =Info{1,3}:h_X:Info{1,4};

h_Y =(Info{1,6}-Info{1,5})/(Pts_Per_Dimension-1); 
Y   =Info{1,5}:h_X:Info{1,6};

h_Z =(Info{1,8}-Info{1,7})/(Pts_Per_Dimension-1); 
Z   =Info{1,7}:h_X:Info{1,8};

% F_Output=zeros(Pts_Per_Dimension, Pts_Per_Dimension, Pts_Per_Dimension, timesteps);
timesteps = length(Info);

%%
fileID = fopen('../SIMULATION_OUTPUT/Mean_PDFs_0.bin');
Data=fread(fileID,[Pts_Per_Dimension^3,timesteps],'float');
fclose(fileID);

%%
[x,y,z] = meshgrid(X,Y,Z);

F_Output = reshape(Data,[Pts_Per_Dimension Pts_Per_Dimension Pts_Per_Dimension timesteps]);

%%
skip = 4;
Integral_vals = zeros(1,floor(timesteps/skip));
iso_val       = zeros(floor(timesteps/skip),2);
Marg_X        = zeros(Pts_Per_Dimension,1);
Marg_Y        = Marg_X;
Marg_Z        = Marg_X;

for l = 1:skip:timesteps
    fprintf(['At step: ',num2str(l),newline]);

    % COMPUTE MARGINAL DENSITIES
    Marg_X = h_Y*h_Z*reshape(sum(F_Output(:,:,:,l),[2 3]),[Pts_Per_Dimension,1]);
    Marg_Y = h_X*h_Z*reshape(sum(F_Output(:,:,:,l),[1 3]),[Pts_Per_Dimension,1]);
    Marg_Z = h_X*h_Y*reshape(sum(F_Output(:,:,:,l),[1 2]),[Pts_Per_Dimension,1]);

    Integral_vals(l)=sum(F_Output(:,:,:,l),[1,2,3])*h_X*h_Y*h_Z;

    % COMPUTE THE "HEIGHT" OF THE PDF OVER WHOM WE HAVE "conf_lvl" MASS
    iso_val(l,:)=ComputeRegion(Data(:,l)',conf_lvl*Integral_vals(l),h_X,DIMENSIONS);

    figure(l)
    xlim([0,1]);xlabel(Name_var2);
    ylim([0,1]);ylabel(Name_var1);
    zlim([0,1]);zlabel(Name_var3);
    isosurface(x,y,z,F_Output(:,:,:,l),iso_val(l,1));view(3);colorbar;grid on; grid minor; lightangle(-45,45); drawnow;

    figure(100+l)
    subplot(3,1,1)
    plot(X,Marg_X,'r.-');xlabel(Name_var1);ylabel('Probability Density'); drawnow;
    subplot(3,1,2)
    plot(Y,Marg_Y,'r.-');xlabel(Name_var2);ylabel('Probability Density'); drawnow;
    subplot(3,1,3)
    plot(Z,Marg_Z,'r.-');xlabel(Name_var3);ylabel('Probability Density'); drawnow;
    pause(1);
end
