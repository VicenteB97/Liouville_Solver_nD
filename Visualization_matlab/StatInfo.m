function Stats = StatInfo(f,X,Y,h)

Int=sum(f,[1,2])*h^2; % Total mass (or total integral of the PDF given by simulation)

% Mean values
for j=1:length(Y)
    w(:,j)=X(:).*f(:,j);
end
for i=1:length(X)
    v(i,:)=Y(:)'.*f(i,:);
end
% Mean1=sum(w,[1,2]).*h^2.*((X(end)-X(1))*(Y(end)-Y(1)));
% Mean2=sum(v,[1,2]).*h^2.*((X(end)-X(1))*(Y(end)-Y(1)));
Mean1=sum(w,[1,2]).*h^2;
Mean2=sum(v,[1,2]).*h^2;

% Variance
for j=1:length(Y)
    u(:,j)=( (X(:)-Mean1).^2+(Y(j)-Mean2).^2 ).*f(:,j);
end
Var=sum(u,[1,2]).*h^2;

% Covariance
for j=1:length(Y)
    z(:,j)=( (X(:)-Mean1)*(Y(j)-Mean2) ).*f(:,j);
end
% Cov=sum(z,[1,2]).*h^2.*((X(end)-X(1))*(Y(end)-Y(1)));
Cov=sum(z,[1,2]).*h^2;

Stats(1)=Int;
Stats(2)=Mean1;
Stats(3)=Mean2;
Stats(4)=sqrt(Var);
Stats(5)=Cov;
end

