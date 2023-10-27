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

F_Test_2 = F_Output(:,:,1:2:end); clear F_Output
Log_2 = Log; clear Log
t_2 = t; clear t

% Read 3 and change name
load("VanDerPol_Test_3.mat")

F_Test_3 = F_Output; clear F_Output
Log_3 = Log; clear Log
t_3 = t; clear t


%% Compare the norm of the solutions
[temp_pts,~,temp_steps] = size(F_Test_1);
Pts_per_Dim = temp_pts;

Show_error_sheet = false;

Error_PDF = zeros(temp_pts,temp_pts);

X_min = -5.5; X_max = 5.5; Y_min = -5.5; Y_max = 5.5;
X=X_min:(X_max-X_min)/(Pts_per_Dim - 1):X_max;
Y=Y_min:(Y_max-Y_min)/(Pts_per_Dim - 1):Y_max;

temp_diff_L1 = zeros(1,length(t_1));
temp_diff_L2 = zeros(1,length(t_1));
temp_diff_LInf = zeros(1,length(t_1));

% Compute errors in test errors 1-2
for k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Test_1(:,:,k) - F_Test_2(:,:,k)),[1,2])*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_L2(k) = sqrt(sum((F_Test_1(:,:,k) - F_Test_2(:,:,k)).^2,[1,2]))*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_LInf(k) = max(abs(F_Test_1(:,:,k) - F_Test_2(:,:,k)),[],'all');
end

L1_Test_12 = sum(temp_diff_L1) * (t_1(2)-t_1(1));
L2_Test_12 = sum(temp_diff_L2) * (t_1(2)-t_1(1));
LInf_Test_12 = max(temp_diff_LInf);

%%
figure(1)
plot(t_1, temp_diff_L1,t_1,temp_diff_L2); legend('L_1','L_2','L_{Inf}'); grid minor

figure(10)
title('Error comparison')
error_step = 210;
Error_PDFs(:,:) = F_Test_1(:,:,error_step) - F_Test_2(:,:,error_step);
if 1
    subplot(1,2,1)
    mesh(X,Y,Error_PDFs); colorbar;view(0,90); %
    xlabel('Position'); ylabel('Velocity');pause(0.2)
end

error_step = 250;
Error_PDFs(:,:) = F_Test_1(:,:,error_step) - F_Test_2(:,:,error_step);
if 1
    subplot(1,2,2)
    mesh(X,Y,Error_PDFs); colorbar;view(0,90); %./max(F_Test_2(:,:,error_step),[],'all')
    xlabel('Position'); ylabel('Velocity');pause(0.2)
end


% Compute errors in test errors 1-3
for k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Test_1(:,:,k) - F_Test_3(:,:,k)),[1,2])*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_L2(k) = sqrt(sum((F_Test_1(:,:,k) - F_Test_3(:,:,k)).^2,[1,2]))*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_LInf(k) = max(abs(F_Test_1(:,:,k) - F_Test_3(:,:,k)),[],'all');

    Error_PDFs(:,:) = F_Test_1(:,:,k) - F_Test_3(:,:,k);
    if Show_error_sheet
        figure(2)
        imagesc(X,Y,Error_PDFs); colorbar
        pause(0.2)
    end
end

L1_Test_13 = sum(temp_diff_L1) * (t_1(2)-t_1(1));
L2_Test_13 = sum(temp_diff_L2) * (t_1(2)-t_1(1));
LInf_Test_13 = max(temp_diff_LInf);

figure(3)
plot(t_1, temp_diff_L1,t_1,temp_diff_L2,t_1, temp_diff_LInf); legend('L_1','L_2','L_{Inf}');grid minor

% Compute errors in test errors 2-3
for k = 1:temp_steps
    temp_diff_L1(k) = sum(abs(F_Test_2(:,:,k) - F_Test_3(:,:,k)),[1,2])*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_L2(k) = sqrt(sum((F_Test_2(:,:,k) - F_Test_3(:,:,k)).^2,[1,2]))*(X(2)-X(1))*(Y(2)-Y(1));
    temp_diff_LInf(k) = max(abs(F_Test_2(:,:,k) - F_Test_3(:,:,k)),[],'all');

    Error_PDFs(:,:) = F_Test_2(:,:,k) - F_Test_3(:,:,k);
    if Show_error_sheet
        figure(2)
        imagesc(X,Y,Error_PDFs); colorbar
        pause(0.2)
    end
end

L1_Test_23 = sum(temp_diff_L1) * (t_1(2)-t_1(1));
L2_Test_23 = sum(temp_diff_L2) * (t_1(2)-t_1(1));
LInf_Test_23 = max(temp_diff_LInf);

figure(5)
plot(t_1, temp_diff_L1,t_1,temp_diff_L2,t_1, temp_diff_LInf); legend('L_1','L_2','L_{Inf}');grid minor

%% Generate graphics in PDF