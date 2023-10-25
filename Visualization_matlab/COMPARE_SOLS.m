clear
close all
clc

%% Read solutions (previously read from the simulation outputs)
% Read 1 and change name
load("VanDerPol_Test_1.mat")

F_Test_1 = F_Output; clear F_Output
Log_1 = Log; clear Log
t_1 = t; clear t

% Read 2 and change name
load("VanDerPol_Test_2.mat")

F_Test_2 = F_Output; clear F_Output
Log_2 = Log; clear Log
t_2 = t; clear t

% Read 3 and change name
load("VanDerPol_Test_1.mat")

F_Test_3 = F_Output; clear F_Output
Log_3 = Log; clear Log
t_3 = t; clear t


%% Compare the norm of the solutions
[temp_pts,~,temp_steps] = size(F_Test_1);
Pts_per_Dim = temp_pts;

X_min = -5.5; X_max = 5.5; Y_min = -5.5; Y_max = 5.5;
X=X_min:(X_max-X_min)/(Pts_per_Dim - 1):X_max;
Y=Y_min:(Y_max-Y_min)/(Pts_per_Dim - 1):Y_max;

temp_diff_L1 = zeros(1,length(t_1));
temp_diff_L2 = zeros(1,length(t_1));
temp_diff_LInf = zeros(1,length(t_1));

% Compute errors in test errors 1-2
parfor k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Test_1(:,:,k) - F_Test_2(:,:,2*(k-1) + 1)),[1,2]);
    temp_diff_L2(k) = sum(abs(F_Test_1(:,:,k) - F_Test_2(:,:,2*(k-1) + 1)).^2,[1,2]);
    temp_diff_LInf(k) = max(abs(F_Test_1(:,:,k) - F_Test_2(:,:,2*(k-1) + 1)),[1,2]);
end

L1_Test_12 = sum(temp_diff_L1);
L2_Test_12 = sum(temp_diff_L2);
LInf_Test_12 = sum(temp_diff_LInf);

% Compute errors in test errors 1-3
parfor k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Test_1(:,:,k) - F_Test_3(:,:,k)),[1,2]);
    temp_diff_L2(k) = sum(abs(F_Test_1(:,:,k) - F_Test_3(:,:,k)).^2,[1,2]);
    temp_diff_LInf(k) = max(abs(F_Test_1(:,:,k) - F_Test_3(:,:,k)),[1,2]);
end

L1_Test_13 = sum(temp_diff_L1);
L2_Test_13 = sum(temp_diff_L2);
LInf_Test_13 = sum(temp_diff_LInf);

% Compute errors in test errors 2-3
parfor k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Test_2(:,:,2*(k-1) + 1) - F_Test_3(:,:,k)),[1,2]);
    temp_diff_L2(k) = sum(abs(F_Test_2(:,:,2*(k-1) + 1) - F_Test_3(:,:,k)).^2,[1,2]);
    temp_diff_LInf(k) = max(abs(F_Test_2(:,:,2*(k-1) + 1) - F_Test_3(:,:,k)),[1,2]);
end

L1_Test_23 = sum(temp_diff_L1);
L2_Test_23 = sum(temp_diff_L2);
LInf_Test_23 = sum(temp_diff_LInf);


% Compare with different reinitialization timestep (double them or make them half the size)

% Compare with several different RBF support radius (3/4 different, some will not appear)

%% Generate graphics in PDF