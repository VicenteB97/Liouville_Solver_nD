clear
close all
clc
set(0,'defaulttextinterpreter','latex')
delete(gcp('nocreate'));

%% PRE-(POST-PROCESSING)
Show_AMR = true;
Show_Confidence_Region = true;
Show_Animation = false;
Save_Animation = false; % STILL NOT WORKING...
Save_infoFile = true;

Sim_infoFile_name = "SIR.mat";

Name_var1 = 'S';
Name_var2 = 'I';
Name_var3 = 'R';

PHASE_SPACE_DIMENSIONS=3;
conf_lvl = 0.95;


%% CHANGE FOR YOUR CURRENT COMPUTER
% best in S: 4.7
% best overall 5.3
% Info=readcell  ('../output/SIR_r_5y3_reinit_2_samples_32_Simulation_info_0.csv');
% fileID = fopen('../output/SIR_r_5y3_reinit_2_samples_32_Mean_PDFs_0.bin');

Info=readcell  ('../output/SIR_r_6y3_reinit_2_samples_8_high_Simulation_info_0.csv');
fileID = fopen('../output/SIR_r_6y3_reinit_2_samples_8_high_Mean_PDFs_0.bin');



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

Data=fread(fileID,[Pts_Per_Dimension^3,timesteps],'float');
fclose(fileID);

%%
[x,y,z] = meshgrid(X,Y,Z);

F_Output = reshape(Data,[Pts_Per_Dimension Pts_Per_Dimension Pts_Per_Dimension timesteps]);

if Save_infoFile
    save(Sim_infoFile_name)
end

%%
skip = 1;
Integral_vals = zeros(1,floor(timesteps/skip));
iso_val       = zeros(floor(timesteps/skip),2);
Marg_X        = zeros(Pts_Per_Dimension,1);
Marg_Y        = Marg_X;
Marg_Z        = Marg_X;

ttss = ceil(8/30*(timesteps))
output_name = ['../output/liouville_SIR_r_6y3_reinit_2_samples_32_time_', num2str(ttss-1), '.pdf'];
output_name_ci = ['../output/liouville_SIR_ci_r_6y3_reinit_2_samples_32_time_', num2str(ttss-1), '.pdf'];
for l = ttss:ttss
    fprintf(['At step: ',num2str(l),newline]);

    % COMPUTE MARGINAL DENSITIES
    Marg_X = h_Y*h_Z*reshape(sum(F_Output(:,:,:,l),[2 3]),[Pts_Per_Dimension,1]);
    Marg_Y = h_X*h_Z*reshape(sum(F_Output(:,:,:,l),[1 3]),[Pts_Per_Dimension,1]);
    Marg_Z = h_X*h_Y*reshape(sum(F_Output(:,:,:,l),[1 2]),[Pts_Per_Dimension,1]);

    Integral_vals(l)=sum(F_Output(:,:,:,l),[1,2,3])*h_X*h_Y*h_Z;

    % COMPUTE THE "HEIGHT" OF THE PDF OVER WHOM WE HAVE "conf_lvl" MASS
    iso_val(l,:)=ComputeRegion(Data(:,l)',conf_lvl*Integral_vals(l),h_X,PHASE_SPACE_DIMENSIONS);

    figure(l);
    xlim([0,1]);xlabel(Name_var2);
    ylim([0,1]);ylabel(Name_var1);
    zlim([0,1]);zlabel(Name_var3);
    title(['L: SIR confidence surface at $t = ', num2str(Info{2,l}), '$'], 'FontSize', 16);
    isosurface(x,y,z,F_Output(:,:,:,l),iso_val(l,1));view(3);colorbar;
    grid on; grid minor; lightangle(-45,45); drawnow;
    exportgraphics(gcf,output_name_ci,"Resolution",300,'ContentType','vector',...
        'BackgroundColor','none')

    figure(2);
    hold on;
    grid on;
    xlim([0,1]);
     ylim([0,35]);
    %     subplot(3,1,1)
    plot(X,Marg_X,'r-');
    plot(Y,Marg_Y,'g--');
    plot(Z,Marg_Z,'b.-');
    legend("S","I","R")
    title( ['L: SIR per-component PDFs at $t =  ', num2str(Info{2,l}), '$'],'FontSize', 16);
    hold off;
    exportgraphics(gcf,output_name,"Resolution",300,'ContentType','vector',...
        'BackgroundColor','none')
end