% contour plot on regular grid
ymax=Ny*h;

% computation of contour levels
val_max=0;
val_min=0;
for i=1:Nx
    for j=1:Ny
        if(P(j,i)<val_min)
            val_min=P(j,i);
        end
        if(P(j,i)>val_max)
            val_max=P(j,i);
        end
    end
end

Nlin=100;
step=(val_max-val_min)/(Nlin-1);

for i=1:Nlin
    v(i)=val_min+(i-1)*step;
end

for i=1:Nx-1

    if(i>=m1 & i<m2)
        ymax=ymax-h;
    end
    
    xcp(1)=xp(i);
    xcp(2)=xp(i+1);
    clear ycp;
    clear Pcp;
    for j=1:Ny
        if(yp(j)<=(ymax+h))
        % if(yp(j)>ymax)
            ycp(j)=yp(j);
            Pcp(j,1)=P(j,i);
            Pcp(j,2)=P(j,i+1);
        end
    end
    contour(xcp,ycp,Pcp,v);
    % contourf(xcp,ycp,Pcp,v);
end

colorbar('North','limits',[0,0.5]);
% colorbar('North');e
% colorbar('CLimMode','manual');
% colorbar('CLim',[0,0.005]);
text(0.24,0.10,['time = ',num2str(time),' s']);

