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

%% CHANGE FOR YOUR CURRENT COMPUTER

Info=readcell  ('../SIMULATION_OUTPUT/Simulation_Info_6.csv');
Data=readmatrix('../SIMULATION_OUTPUT/Mean_PDFs_6.csv');

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
xslice = [];    % location of y-z planes
yslice = [0:.05:1];    % location of x-z plane
zslice = [0:.05:1];    % location of x-y planes

timesteps = length(Data)/Total_Pts;
skip = 2;
Integral_vals = zeros(1,floor(timesteps/skip));
iso_val       = zeros(floor(timesteps/skip),2);

for l = 1:skip:timesteps
    fprintf(['At step: ',num2str(l),newline]);
    parfor k=1:Pts_Per_Dimension
        for j=1:Pts_Per_Dimension
            for i=1:Pts_Per_Dimension
                i_aux=i+(j-1)*Pts_Per_Dimension+(k-1)*Pts_Per_Dimension^2+(l-1)*Total_Pts;
                F_Output(i,j,k) = Data(1,i_aux);
            end
        end
    end

    Integral_vals(l)=sum(F_Output,'all')*h_X*h_Y*h_Z;

    iso_val(l,:)=ComputeRegion(Data(1,Pts_Per_Dimension^DIMENSIONS*(l-1)+1:Pts_Per_Dimension^DIMENSIONS*l),0.95*Integral_vals(l),h_X,DIMENSIONS);

    figure(l)
    xlim([0,1]);xlabel(Name_var1);
    ylim([0,1]);ylabel(Name_var2);
    zlim([0,1]);zlabel(Name_var3);
%     contourslice(x,y,z,F_Output,xslice,yslice,zslice,10,'linear');view(3);drawnow;colorbar;
    isosurface(x,y,z,F_Output,iso_val(l,1));view(3);colorbar;grid on; grid minor; lightangle(-45,45); drawnow;
    pause(1);
end
