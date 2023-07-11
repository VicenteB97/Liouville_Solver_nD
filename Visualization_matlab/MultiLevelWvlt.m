function [MeshId] = MultiLevelWvlt(f0,FineLvl,CoarseLvl,tol)

% We pass the PDF and the level and tolerance info. We return the ID; that
% is, (i,j), of the grid points selected

%% Initialization of loop
[SS,SD,DS,DD]=TwoDimWavelet(f0);

% Grid Generation
MeshId=[];
[m,n]=size(SS);

for i=1:m
    for j=1:n
        
        % We add the checking of emptiness of the MeshId vector because we
        % don't really know which one is going to be added first
        if abs(SD(i,j))>tol % Points at the right of the approx
            if isempty(MeshId)
                MeshId=[2*i,2*j-1,1];
            else
                MeshId=vertcat(MeshId,[2*i,2*j-1,1]);
            end
        end
        if abs(DS(i,j))>tol % Points upwards
            if isempty(MeshId)
                MeshId=[2*i-1,2*j,1];
            else
                MeshId=vertcat(MeshId,[2*i-1,2*j,1]);
            end
        end
        if abs(DD(i,j))>tol % Points at the diagonal
            if isempty(MeshId)
                MeshId=[2*i,2*j,1];
            else
                MeshId=vertcat(MeshId,[2*i,2*j,1]);
            end            
        end
        
    end
end

%% General loop
for k=2:FineLvl-CoarseLvl
    
[SS,SD,DS,DD]=TwoDimWavelet(SS);
% lvl = 1;
% [SS,SD,DS,DD] = lwt2(SS,'Wavelet',"bior2.8",'Level',lvl);
% SD=cell2mat(SD);
% DS=cell2mat(DS);
% DD=cell2mat(DD);

    % Grid Generation
    [m,n]=size(SS);
    for i=1:m
        for j=1:n
            if abs(SD(i,j))>tol % Points at the right of the approx
                if isempty(MeshId)
                    MeshId=[2^k*i+1-2^(k-1),2^k*j-2^(k-1),k];
                else
                    MeshId=vertcat(MeshId,[2^k*i+1-2^(k-1),2^k*j-2^(k-1),k]);
                end
            end
            if abs(DS(i,j))>tol % Points upwards
                if isempty(MeshId)
                    MeshId=[2^k*i-2^(k-1),2^k*j+1-2^(k-1),k];
                else
                    MeshId=vertcat(MeshId,[2^k*i-2^(k-1),2^k*j+1-2^(k-1),k]);
                end
            end
            if abs(DD(i,j))>tol % Points at the diagonal
                if isempty(MeshId)
                    MeshId=[2^k*i+1-2^(k-1),2^k*j+1-2^(k-1),k];
                else
                    MeshId=vertcat(MeshId,[2^k*i+1-2^(k-1),2^k*j+1-2^(k-1),k]);
                end
            end
        end
    end
end

%% Add the grid points at the coarsest level
for i=1:m
    for j=1:n
        MeshId=vertcat(MeshId,[2^k*i-2^(k-1),2^k*j-2^(k-1),k]);
    end
end

end

