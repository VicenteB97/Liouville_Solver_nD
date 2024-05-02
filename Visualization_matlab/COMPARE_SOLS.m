clear
close all
clc

test_name_1 = "VdP_2000";
test_name_2 = "VdP 200";

% Load info for first test:
load(test_name_1 + ".mat")
F_Output_1 = F_Output; 
MargX_1 = MargX;
MargY_1 = MargY;
Stats_1D_X_1 = Stats_1D_X;
Stats_1D_Y_1 = Stats_1D_Y;

load(test_name_2 + ".mat")
F_Output_2 = F_Output; clear F_Output;
MargX_2 = MargX; clear MargX;
MargY_2 = MargY; clear MargY;
Stats_1D_X_2 = Stats_1D_X;
Stats_1D_Y_2 = Stats_1D_Y;

[temp_pts,~,temp_steps] = size(F_Output_1);
Pts_per_Dim = temp_pts;

X_min = Info{1,3}; X_max = Info{1,4}; Y_min = Info{1,5}; Y_max = Info{1,6};
X=X_min:(X_max-X_min)/(Info{1,2} - 1):X_max;
Y=Y_min:(Y_max-Y_min)/(Info{1,2} - 1):Y_max;

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
plot(t, temp_diff_L1,t,temp_diff_L2); legend('L_1','L_2','L_{Inf}');

% figure(2)
% title('Error comparison')
% error_step = 125;
% Error_PDFs(:,:) = F_Output_1(:,:,error_step) - F_Output_2(:,:,error_step);
% if 1
% %     subplot(1,2,1)
%     mesh(X,Y,Error_PDFs); colorbar;view(0,90); %
%     xlabel('Position'); ylabel('Velocity');pause(0.2)
% end
%% Generate graphics in PDF
figure(3)
subplot(1,2,1)
plot(t,Stats_1D_X_1(:,2),'b.-',t,Stats_1D_X_2(:,2),'r.-'); 
legend(test_name_1,test_name_2); xlabel('t');ylabel('Position')
subplot(1,2,2)
plot(t,Stats_1D_Y_1(:,2),'b.-',t,Stats_1D_Y_2(:,2),'r.-'); 
legend(test_name_1,test_name_2);xlabel('t');ylabel('Velocity')

figure(4)
for k=1:length(t)
    subplot(1,2,1)
    plot(Y,MargX_1(:,k),'b-.',Y,MargX_2(:,k),'r-.');
    xlabel('X (Position)')
    legend(test_name_1,test_name_2)

    subplot(1,2,2)
    plot(Y,MargY_1(:,k),'b-.',Y,MargY_2(:,k),'r-.'); 
    legend(test_name_1,test_name_2); 
    xlabel('DX (Velocity)');
    
    title(['Time: ', num2str(t(k))]); pause(0.5)
end