function output = Stats(pdf_list,DomX,confidenceLvl)

output=zeros(length(pdf_list(1,:)),6);
%% 1: The total mass
output(:,1)=sum(pdf_list(:,:),1)*(DomX(2)-DomX(1));

%% 2: Mean
output(:,2)=sum(pdf_list(:,:).*DomX(:),1)./sum(pdf_list(:,:),1);%*(DomX(2)-DomX(1));

%% 3: Standard Deviation
for k=1:length(pdf_list(1,:))
    aux=0;
    for i=1:length(DomX)
        aux=aux+sum(pdf_list(i,k).*(DomX(i)-output(k,2)').^2,1)*(DomX(2)-DomX(1));
    end
    output(k,3)=sqrt( aux );
end

N_X_refined = 2^13;
DomX_refined = DomX(1):1/(N_X_refined + 1):DomX(end);

%% 4: Confidence region (REHINK THIS!!!!!)
for k=1:length(pdf_list(1,:))

    pdf_list_refined = interp1(DomX,pdf_list(:,k),DomX_refined);
    % Compute the confidence region for each time step shown in graphics %%
    confidenceLvl = confidenceLvl * (output(k,1));

    if confidenceLvl<1
        sortedPDF=sort(pdf_list_refined,'descend');
        output(k,4)=sortedPDF(1);
    
        lambda_0=0;
        phi_0=1;
    
        lambda_1=sortedPDF(1);
        phi_1=0;
    
        idx=1;
        i_new=1;
        
        tol=5e-3;
        
        MaxIts=1000000;
        Its=0;
        
        while Its<MaxIts % to make sure there is no overflow due to slow convergence
            
            lambda_c=0.5*(lambda_0+lambda_1);
            
            idx=i_new;
            while sortedPDF(idx)>=lambda_c && idx<length(sortedPDF)
                idx=idx+1;
            end
    
        
            phi_c=phi_1+(DomX_refined(2)-DomX_refined(1))*sum(sortedPDF(i_new:idx));
        
            error=phi_c-confidenceLvl;
        
            if error<0 && abs(error)>tol
                lambda_1=lambda_c;
                phi_1=phi_c;
                i_new=idx;
                Its=Its+1;
        
            elseif error>0 && abs(error)>tol
                lambda_0=lambda_c;
                Its=Its+1;
            elseif abs(error)<=tol
                output(k,4)=sortedPDF(idx);
                break;
            elseif Its==MaxIts-1
                output(k,4)=sortedPDF(idx);
            end
        end
        % Now, we compute the interval boundaries of the confidence region
        u=[];
        u=find(pdf_list_refined>=output(k,4));
        output(k,5:6)=[DomX_refined(min(u)),DomX_refined(max(u))];
    else
        % Now, we compute the interval boundaries of the confidence region
        u=[];
        u=find(pdf_list_refined>=1e-7);
        output(k,5:6)=[DomX_refined(min(u)),DomX_refined(max(u))];
    end
end % set a breakpoint HERE to see the integral computation
end

