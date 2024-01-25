close all
clear
clc

%% Choose simulations to compare
Solution1_fileName = 'SIR_1_8.mat';
Solution2_fileName = 'SIR_2_4.mat';

%% Load the simulation information for the first case
Solution1_data = load(Solution1_fileName);

Solution1_info = Solution1_data.Info;
infoSize = size(Solution1_info);

Solution1_timeInstants = zeros(1,infoSize(2));
for i = 1:infoSize(2)
    Solution1_timeInstants(i) = Solution1_info{2,i};
end

% The PDFs are stored in Solution1_data.F_Output...think about it


%% Load the simulation information for the first case
Solution2_data = load(Solution2_fileName);

Solution2_info = Solution2_data.Info;
infoSize = size(Solution2_info);

Solution2_timeInstants = zeros(1,infoSize(2));
for i = 1:infoSize(2)
    Solution2_timeInstants(i) = Solution2_info{2,i};
end

% The PDFs are stored in Solution2_data.F_Output...think about it

%% Error Checking
if(Solution1_data.timesteps ~= Solution2_data.timesteps)
    error('Not equal timesteps');
end
timeInstants = Solution1_timeInstants;

if(Solution1_data.Pts_Per_Dimension ~= Solution2_data.Pts_Per_Dimension)
    error('Different spatial resolution');
end
ptsDimension = Solution1_data.Pts_Per_Dimension;

if(Solution1_data.PHASE_SPACE_DIMENSIONS ~= Solution2_data.PHASE_SPACE_DIMENSIONS)
    error('Different dimensionality')
end
dimensions = Solution1_data.PHASE_SPACE_DIMENSIONS;

% The spatial domain
X = Solution1_data.X;
Y = Solution1_data.Y;
Z = Solution1_data.Z;

%% Join everything
errorManifold = zeros(ptsDimension,ptsDimension,ptsDimension);
l1_errorCurve = zeros(1,length(timeInstants));
l2_errorCurve = zeros(1,length(timeInstants));
lInf_errorCurve = zeros(1,length(timeInstants));

for i = 1:length(timeInstants)
    errorManifold = abs(Solution1_data.F_Output(:,:,:,i) - Solution2_data.F_Output(:,:,:,i));

    lInf_errorCurve(i)  = max(errorManifold,[],'all');
    l1_errorCurve(i)    = sum(errorManifold,'all');
    l2_errorCurve(i)    = sqrt(sum(errorManifold.^2,'all'));

    figure(1)
    plot(X,Solution1_data.Marg_X(:),X,Solution2_data.Marg_X(:));legend('Sol1','Sol2');title(['Time: ', num2str(timeInstants(i))]);pause(1)
    figure(2)
    plot(Y,Solution1_data.Marg_Y(:),Y,Solution2_data.Marg_Y(:));legend('Sol1','Sol2');title(['Time: ', num2str(timeInstants(i))]);pause(1)
    figure(3)
    plot(Z,Solution1_data.Marg_Z(:),Z,Solution2_data.Marg_Z(:));legend('Sol1','Sol2');title(['Time: ', num2str(timeInstants(i))]);pause(1)

end

%%
figure(4)
plot(timeInstants,lInf_errorCurve)
figure(5)
plot(timeInstants,l1_errorCurve)
figure(6)
plot(timeInstants,l2_errorCurve)