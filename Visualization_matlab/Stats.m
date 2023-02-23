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

%% 4: Confidence region (REHINK THIS!!!!!)
for k=1:length(pdf_list(1,:))

    % Compute the confidence region for each time step shown in graphics %%
    confidenceLvl = confidenceLvl;
    confidenceLvl = confidenceLvl * (output(k,1));

    if confidenceLvl<1
        sortedPDF=sort(pdf_list(:,k),'descend');
        output(k,4)=sortedPDF(1);
    
        lambda_0=0;
        phi_0=1;
    
        lambda_1=sortedPDF(1);
        phi_1=0;
    
        idx=1;
        i_new=1;
        
        tol=1e-2;
        
        MaxIts=100000;
        Its=0;
        
        while Its<100000 % to make sure there is no overflow due to slow convergence
            
            lambda_c=0.5*(lambda_0+lambda_1);
            
            idx=i_new;
            while sortedPDF(idx)>=lambda_c && idx<length(sortedPDF)
                idx=idx+1;
            end
    
        
            phi_c=phi_1+(DomX(2)-DomX(1))*sum(sortedPDF(i_new:idx));
        
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
                output(k,4)=sortedPDF(1);
            end
        end
    % Now, we compute the interval boundaries of the confidence region
    u=[];
    u=find(pdf_list(:,k)>=output(k,4)*0.9);
    output(k,5:6)=[DomX(min(u)),DomX(max(u))];
else
    % Now, we compute the interval boundaries of the confidence region
    u=[];
    u=find(pdf_list(:,k)>=1e-6);
    output(k,5:6)=[DomX(min(u)),DomX(max(u))];
end

end % set a breakpoint HERE to see the integral computation
end

