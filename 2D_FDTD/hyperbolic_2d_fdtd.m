% function parabolic_2d_fdtd(Niter)
%FDTD for Parabolic PDE - 2D

% material
mur=1.; % relative permeability
epsilonr=1.; % relative permitivity
mu0=4.*pi*1e-7; % permeability of vacuum
epsilon0=8.85e-12; % permitivity of vacuum
f=2.7e9; % frequency
period=1/f; % period
omega=2.*pi*f; % circular frequency

mu=mur*mu0;
epsilon=epsilonr*epsilon0;
v=1/sqrt(mu*epsilon); % EMW velocity

lambda=period*v; % wavelength

delta_t_max=h/(v*sqrt(2)); % maximum allowed time step according to stability

% r=0.8*1/2; % stability condition

% delta_t=sqrt(r)*h/v; % time step

delta_t=0.8*delta_t_max;

r_ep=delta_t/(h*epsilon);
r_mu=delta_t/(h*mu);
r=(delta_t*v/h)^2;

% iterations through the time
% Niter=55;
for i=1:Niter % loop ever the time
   for j=1:Ng % loop over the grid
       if(type_bc(j)==0) % regular point
           
           Hy_Left=Hy(j,1,i);
           Hy_Right=Hy(j,2,i);
           Hx_Bottom=Hx(j,1,i);
           Hx_Top=Hx(j,2,i);
           
           E_Right=E(neighbors(j,2),i);
           E_Top=E(neighbors(j,4),i);
           E_Center=E(j,i);
      
           % intermediate magnetic field values
           Hx(j,2,i+1)=Hx_Top+r_mu*(E_Center-E_Top);
           Hy(j,2,i+1)=Hy_Right+r_mu*(E_Right-E_Center);
           % Hx(neighbors(j,3),1,i+1)=Hx(j,2,i+1);
           % Hy(neighbors(j,1),1,i+1)=Hy(j,2,i+1);
           Hx(neighbors(j,4),1,i+1)=Hx(j,2,i+1);
           Hy(neighbors(j,2),1,i+1)=Hy(j,2,i+1);
           
           Hy_Left=Hy(j,1,i+1);
           Hy_Right=Hy(j,2,i+1);
           Hx_Bottom=Hx(j,1,i+1);
           Hx_Top=Hx(j,2,i+1);
           
           E(j,i+1)=E_Center+r_ep*(Hy_Right-Hy_Left-Hx_Top+Hx_Bottom);
           
       end
       if(type_bc(j)==1) % Dirichlet bc
            E(j,i+1)=value_bc(j); % Ez value for Dirichlet BC

            if(neighbors(j,4)>0 & neighbors(j,2)>0)
            Hy_Left=Hy(j,1,i);
            Hy_Right=Hy(j,2,i);
            Hx_Bottom=Hx(j,1,i);
            Hx_Top=Hx(j,2,i);
           
            E_Right=E(neighbors(j,2),i);
            E_Top=E(neighbors(j,4),i);
            E_Center=E(j,i);
      
            % intermediate magnetic field values
            Hx(j,2,i+1)=Hx_Top+r_mu*(E_Center-E_Top);
            Hy(j,2,i+1)=Hy_Right+r_mu*(E_Right-E_Center);
            Hx(neighbors(j,4),1,i+1)=Hx(j,2,i+1);
            Hy(neighbors(j,2),1,i+1)=Hy(j,2,i+1);
            end
       end
       if(type_bc(j)==3) % ABC
          if(x(j)==h) % input port
              Right=E(neighbors(j,2),i);
              Bottom=E(neighbors(j,3),i);
              Top=E(neighbors(j,4),i);
              Center=E(j,i);
              E(j,i+1)=2*(1-2.*r)*Center+r*(Right+Bottom+Top);
              if(i>1)
                  E(j,i+1)=E(j,i+1)-E(j,i-1);
              end
              E(j,i+1)=E(j,i+1)+sqrt(r)*(1+sqrt(r))*Center;
              E(j,i+1)=E(j,i+1)/(1+sqrt(r));
              if(i==1)
                  E(j,i+1)=E(j,i+1)+value_bc(j)*sin(omega*i*delta_t); % add source term
                  % E(j,i+1)=value_bc(j)*sin(omega*i*delta_t); % add source term
              else
                  % E(j,i+1)=E(j,i+1)+value_bc(j)*sin(omega*i*delta_t); % add source term
                  E(j,i+1)=E(j,i+1)+value_bc(j)*sin(omega*i*delta_t)-value_bc(j)*sin(omega*(i-1)*delta_t); % add source term
                  % E(j,i+1)=value_bc(j)*sin(omega*i*delta_t); % add source term
              end
              
              Hy_Left=Hy(j,1,i);
              Hy_Right=Hy(j,2,i);
              Hx_Bottom=Hx(j,1,i);
              Hx_Top=Hx(j,2,i);
           
              E_Right=E(neighbors(j,2),i);
              E_Top=E(neighbors(j,4),i);
              E_Center=E(j,i);
      
              % intermediate magnetic field values
              Hx(j,2,i+1)=Hx_Top+r_mu*(E_Center-E_Top);
              Hy(j,2,i+1)=Hy_Right+r_mu*(E_Right-E_Center);
              % Hx(neighbors(j,3),1,i+1)=Hx(j,2,i+1);
              % Hy(neighbors(j,1),1,i+1)=Hy(j,2,i+1);
              Hx(neighbors(j,4),1,i+1)=Hx(j,2,i+1);
              Hy(neighbors(j,2),1,i+1)=Hy(j,2,i+1);
              
          else % output port
              Left=E(neighbors(j,1),i);
              Bottom=E(neighbors(j,3),i);
              Top=E(neighbors(j,4),i);
              Center=E(j,i);
              E(j,i+1)=2*(1-2.*r)*Center+r*(Left+Bottom+Top);
              if(i>1)
                  E(j,i+1)=E(j,i+1)-E(j,i-1);
              end
              E(j,i+1)=E(j,i+1)+sqrt(r)*(sqrt(r)+1)*Center;
              E(j,i+1)=E(j,i+1)/(1+sqrt(r));
          end
       end
   end
end

step=Niter+1;

for i=1:Nx
    xp(i)=h*i;
    for j=1:Ny
        P(j,i)=0;
    end
end
for j=1:Ny
    yp(j)=j*h;
end

for i=1:Ng
    k=int8(x(i)/h);
    l=int8(y(i)/h);
    P(l,k)=E(i,step);
end

time=step*delta_t;

% contour(xp,yp,P,50);
% contourf(xp,yp,P);
% contourf(x,y,P);
% colorbar('North');

