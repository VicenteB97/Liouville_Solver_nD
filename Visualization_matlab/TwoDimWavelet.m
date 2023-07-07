function [SS,SD,DS,DD] = TwoDimWavelet(Signal2D)

% Obtain approx and details of a 2D signal using the tensor product of 1D
% Haar Wavelets

[m,n]=size(Signal2D);

SS=zeros(m/2,n/2);
SD=SS;
DS=SS;
DD=SS;

for i=1:m/2
    for j=1:n/2
        %Step One, row transform
        SS_aux=1/2*(Signal2D(2*i-1,2*j-1)+Signal2D(2*i,2*j-1));
        SD_aux=Signal2D(2*i-1,2*j-1)-Signal2D(2*i,2*j-1);
        DS_aux=1/2*(Signal2D(2*i-1,2*j)+Signal2D(2*i,2*j));
        DD_aux=Signal2D(2*i-1,2*j)-Signal2D(2*i,2*j);
        
        %Step Two, column & diagonal transform
        SS(i,j)=1/2*(SS_aux+DS_aux);
        SD(i,j)=-1/2*(SD_aux+DD_aux);
        DS(i,j)=-SS_aux+DS_aux;
        DD(i,j)=SD_aux-DD_aux;
    end 
end
end

