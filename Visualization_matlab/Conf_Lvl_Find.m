function output= Conf_Lvl_Find(pdf_list,DomX,confidenceLvl)
output=zeros(length(pdf_list(1,:)),1);

for k=1:length(pdf_list(1,:))
    sortedPDF=sort(pdf_list(:,k),'descend');
    output(k)=sortedPDF(1);

    lambda_0=0;
    phi_0=1;

    lambda_1=sortedPDF(1);
    phi_1=0;

    idx=1;
    i_new=1;
    
    tol=1e-3;
    
    maxIts=0;
    
    while maxIts<25000 % to make sure there is no overflow due to slow convergence
        
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
            maxIts=maxIts+1;
    
        elseif error>0 && abs(error)>tol
            lambda_0=lambda_c;
            maxIts=maxIts+1;
        elseif abs(error)<=tol
            output(k)=sortedPDF(idx);
            break;
        elseif maxIts==25000-1
            output(k)=1e-9;
        end
    end

end % set a breakpoint HERE to see the integral computation
end

