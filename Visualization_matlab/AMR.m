function [MeshId,val,GridPt,f0_Disc] = AMR(f0,LvlFine,LvlCoarse,X,Y,tol)

% Obtain Mesh Id
MeshId=MultiLevelWvlt(f0,LvlFine,LvlCoarse,tol);

%initialize data in the next loop

% create pool if needed
% pool=gcp('nocreate');
% if ~isempty(pool)
%     parpool('threads');
% end

val=zeros(length(X),length(Y));
% Info retrieval from the AMR procedure
for k=1:length(MeshId)
    idx1=MeshId(k,1);
    idx2=MeshId(k,2);

    val(idx1,idx2)=1;

    get_Level(k)=MeshId(k,3);
    GridPt(k,:)=[X(idx1),Y(idx2)];
    f0_Disc(k)=f0(idx1,idx2);
end
end


% 
% function [MeshId,val,GridPt,f0_Disc] = AMR(f0,LvlFine,LvlCoarse,X,Y,tol,N)
% 
% % Obtain Mesh Id
% MeshId=MultiLevelWvlt(f0,LvlFine,LvlCoarse,tol);
% 
% % fprintf(['Data reduction/compression ', num2str(100-length(MeshId(:,1))/N^2*100), ' per cent.\n'])
% 
% val=zeros(N+1,N+1);
% 
% %initialize data in the next loop
% 
% % Info retrieval from the AMR procedure
% for k=1:length(MeshId)
%     idx1(k)=MeshId(k,1);
%     idx2(k)=MeshId(k,2);
%     val(idx1(k),idx2(k))=1;
%     GridPt(k,:)=[X(idx1(k)),Y(idx2(k))];
%     f0_Disc(k)=f0(idx1(k),idx2(k));
% end
% end
% 
