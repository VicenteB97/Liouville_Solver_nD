function Marg = MarginalPlots(f,X,Y,MargX,MargY,varArg,varInit,varFinal,showPlot,lengthT)

% varArg is the one which decides whether we get the first or second
% variable
% varInit and varFinal decide the length of the intervals where we compute
% the marginalization

N=length(X);
h1=X(2)-X(1);
h2=Y(2)-Y(1);

Marg=zeros(N,1);

figure(lengthT+100);

if varArg==2
    
    idxInit=find(Y<varInit,1,'last')+1;
    idxFinal=find(Y>varFinal,1,'first');
    if idxFinal-idxInit>1
        for i=1:N
            Marg(i)=sum(f(i,idxInit:idxFinal))./sum(MargY(idxInit:idxFinal));
        end
    else
        for i=1:N
            Marg(i)=sum(f(i,idxInit:idxFinal))./sum(MargY(idxInit:idxFinal));
        end
    end
    if showPlot
        plot(X,Marg);xlabel('Var1');ylabel('Probability Density');title(['Marginal Density for Var2 in [',num2str(varInit),',',num2str(varFinal),']']);
    end
else

    idxInit=find(X<varInit,1,'last')+1;
    idxFinal=find(X>varFinal,1,'first');
    if idxFinal-idxInit>1
        for i=1:N
            Marg(i)=sum(f(idxInit:idxFinal,i))./sum(MargX(idxInit:idxFinal));
        end
    else
        for i=1:N
            Marg(i)=sum(f(idxInit:idxFinal,i))./sum(MargX(idxInit:idxFinal));
        end
    end
    if showPlot
        plot(Y,Marg);xlabel('Var2');ylabel('Probability Density');title(['Marginal Density for Var1 in [',num2str(varInit),',',num2str(varFinal),']']);
    end
end
end

