function [FinalRegion_bound] = ComputeRegion(pdf,confidenceLvl,Len1,Len2,h)
if confidenceLvl==1
    FinalRegion_bound=1e-9;
else

    %transform the matrix to a vector
    sz_vec=size(pdf);
    f_Disc=zeros(1,sz_vec(1)*sz_vec(2));
    for j=1:sz_vec(2)
        for i=1:sz_vec(1)
            f_Disc(i+sz_vec(1)*j)=pdf(i,j);
        end
    end
    
    % First, we sort the values vector (using the GPU...much faster):
    f_disc_gpu=gpuArray(f_Disc);
    f_disc_gpu=sort(f_disc_gpu,'descend');
    f_Disc=gather(f_disc_gpu);

%     f_Disc=sort(f_Disc,'descend');
    
    tol=0.01;
    % We are given the points as vectors, we are going to try and compute the
    % condidence region as a level set
    IntegralA=1;
    LambdaA=0;
    
    IntegralB=0;
    LambdaB=f_Disc(1,1); % What is the max value of the function?
    i_new=1;
    i_new2=1;
    
    error=1;
    
    maxIts=0;
    
    while (1) % to make sure there is no overflow due to slow convergence
        
        LambdaC=0.5*(LambdaA+LambdaB);
        
        for i=i_new:length(f_Disc)
            if f_Disc(i)>=LambdaC
                i_new2=i;
            end
        end
    
        IntegralC=IntegralB+h^2*sum(f_Disc(i_new:i_new2),["all"]);
    
        error=IntegralC-confidenceLvl;
    
        if maxIts >= 25000
            FinalRegion_bound=1e-9;
            fprintf('Error, max iterations overflow, debug figure or algorithm...\n')
            break;
        elseif error<0 && abs(error)>tol
            LambdaB=LambdaC;
            IntegralB=IntegralC;
            i_new=i_new2;
            maxIts=maxIts+1;
        elseif error>0 && abs(error)>tol
            LambdaA=LambdaC;
            maxIts=maxIts+1;
        elseif abs(error)<=tol || error == 0
            FinalRegion_bound=f_Disc(1,i_new);
            break;
        end
    end
end
end

