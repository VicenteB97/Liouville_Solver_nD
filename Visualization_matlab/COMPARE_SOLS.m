clear
close all
clc

test_1 = 5;
test_2 = 7;

%% Read solutions (previously read from the simulation outputs)
if (test_1 == 1 || test_2 == 1)
    % Read 1 and change name
    load("VanDerPol_Test_1.mat")
    
    F_Test_1 = F_Output; clear F_Output
    Log_1 = Log; clear Log
    t_1 = t; clear t

    % This test has t_reinit = 0.02, with n_samples = 30
end

if (test_1 == 2 || test_2 == 2)
    % Read 2 and change name
    load("VanDerPol_Test_2.mat")
    
    F_Test_2 = F_Output(:,:,1:2:end); clear F_Output
    Log_2 = Log(1:2:end); clear Log
    t_2 = t(1:2:end); clear t

    % This test has t_reinit = 0.01, with n_samples = 30
end

if (test_1 == 3 || test_2 == 3)
    % Read 3 and change name
    load("VanDerPol_Test_3.mat")
    
    F_Test_3 = F_Output; clear F_Output
    Log_3 = Log; clear Log
    t_3 = t; clear t


    % This test has t_reinit = 0.02, with n_samples = 100
end

if(test_1 == 4 || test_2 == 4)
    % Read 4 and change name
    load("VanDerPol_Test_4.mat")
    
    F_Test_4 = F_Output(:,:,1:4:end); clear F_Output
    Log_4 = Log(1:4:end); clear Log
    t_4 = t(1:4:end); clear t

    % This test has t_reinit = 0.005, with n_samples = 30
end

if(test_1 == 5 || test_2 == 5)
    % Read 4 and change name
    load("VanDerPol_Test_5.mat")
    
    F_Test_5 = F_Output(:,:,1:2:end); clear F_Output
    Log_5 = Log(1:2:end); clear Log
    t_5 = t(1:2:end); clear t

    % This test has t_reinit = 0.01, with n_samples = 100
end

if(test_1 == 6 || test_2 == 6)
    % Read 4 and change name
    load("VanDerPol_Test_6.mat")
    
    F_Test_6 = F_Output(:,:,1:4:end); clear F_Output
    Log_6 = Log(1:4:end); clear Log
    t_6 = t(1:4:end); clear t

    % This test has t_reinit = 0.005, with n_samples = 100 
end

if(test_1 == 7 || test_2 == 7)
    % Read 4 and change name
    load("VanDerPol_Test_7.mat")
    
    F_Test_7 = F_Output(:,:,1:4:end); clear F_Output
    Log_7 = Log(1:4:end); clear Log
    t_7 = t(1:4:end); clear t

    % This test has t_reinit = 0.005, with n_samples = 100 
end


%% Compare the norm of the solutions
[L1_error, L2_error, LInf_error] = Compare_PDF(F_Test_5, F_Test_7,t_5);


%% AUXILIARY FUNCTIONS TO BE USED
function [L1_test, L2_test, LInf_test] = Compare_PDF(F_Output_1, F_Output_2,t)

[temp_pts,~,temp_steps] = size(F_Output_1);
Pts_per_Dim = temp_pts;

X_min = -5.5; X_max = 5.5; Y_min = -5.5; Y_max = 5.5;
X=X_min:(X_max-X_min)/(Pts_per_Dim - 1):X_max;
Y=Y_min:(Y_max-Y_min)/(Pts_per_Dim - 1):Y_max;

temp_diff_L1 = zeros(1,length(t));
temp_diff_L2 = zeros(1,length(t));
temp_diff_LInf = zeros(1,length(t));

% Compute errors in test errors 1-2
for k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Output_1(:,:,k) - F_Output_2(:,:,k)),[1,2])*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_L2(k) = sqrt(sum((F_Output_1(:,:,k) - F_Output_2(:,:,k)).^2,[1,2]))*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_LInf(k) = max(abs(F_Output_1(:,:,k) - F_Output_2(:,:,k)),[],'all');
end

L1_test = sum(temp_diff_L1) * (t(2)-t(1));
L2_test = sum(temp_diff_L2) * (t(2)-t(1));
LInf_test = max(temp_diff_LInf);

%%
figure(1)
plot(t, temp_diff_L1,t,temp_diff_L2); legend('L_1','L_2','L_{Inf}'); grid minor

figure(10)
title('Error comparison')
error_step = 210;
Error_PDFs(:,:) = F_Output_1(:,:,error_step) - F_Output_2(:,:,error_step);
if 1
    subplot(1,2,1)
    mesh(X,Y,Error_PDFs); colorbar;view(0,90); %
    xlabel('Position'); ylabel('Velocity');pause(0.2)
end

error_step = 250;
Error_PDFs(:,:) = F_Output_1(:,:,error_step) - F_Output_2(:,:,error_step);
if 1
    subplot(1,2,2)
    mesh(X,Y,Error_PDFs); colorbar;view(0,90); %
    xlabel('Position'); ylabel('Velocity');pause(0.2)
end


end