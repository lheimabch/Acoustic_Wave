% grid definition
clear;

d=0.1;
Nx=70;
Ny=18;
h=d/Ny;

ind=0;
ymax=Ny*h;
d1=(ymax-h)/2.;
Ezamp=1.;

m1=33;
m2=38;

for i=1:Nx
    
    if(i>=m1 & i<m2)
        ymax=ymax-h;
    end
    
    for j=1:Ny
        ind=ind+1;
        x(ind)=i*h;
        y(ind)=j*h;
        
        % type of bc
        type_bc(ind)=0;
        value_bc(ind)=0;
        if(i==1 & j>1 & j<Ny & y(ind) < ymax)
            type_bc(ind)=3; % ABC
            value_bc(ind)=Ezamp*cos((y(ind)-d1-h)*pi/(2*d1)); % value for ABC
        end
        if(i==Nx)
            type_bc(ind)=3; % ABC
            value_bc(ind)=0; % value for ABC
        end
        if(j==1 | j==Ny)
            type_bc(ind)=1; % Dirichlet 
            value_bc(ind)=0;
        end
        
        if(y(ind)>=ymax)
            % if( i>1 & i<Nx)
                type_bc(ind)=1; % Dirichlet
                value_bc(ind)=0;
            % end
            break;
        end        
    end % end - j

end % end - i

% definition of the cells
for i=1:ind
    for j=1:4
        neighbors(ind,j)=0;
    end
end

for j=1:Ny % horizontal neighbors
    yc=j*h;
    ind_p=0;
    for i=1:ind
        if(y(i)==yc)
            if(ind_p>0)
                neighbors(i,1)=ind_p;
                neighbors(ind_p,2)=i;
            end
            ind_p=i;
        end
    end
end
for j=1:Nx % vertical neighbors
    xc=j*h;
    ind_p=0;
    for i=1:ind
        if(x(i)==xc)
            if(ind_p>0)
                neighbors(i,3)=ind_p;
                neighbors(ind_p,4)=i;
            end
            ind_p=i;
        end
    end
end

Ng=ind;

% Zero initial solution
for i=1:Ng
    if(type_bc(i)==1)
        E(i,1)=value_bc(i);
    else
        E(i,1)=0.;
    end
    for j=1:2
        Hx(i,j,1)=0.;
        Hy(i,j,1)=0.;
    end
end
