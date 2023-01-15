% mag. field averaging to nodes

nsteps=200;
mult=0;
for i=1:Ng
   for j=1:nsteps
       
       mult=0;
       if(Hx(i,1,j)~=0)
           Hx_grid(i,j)=Hx_grid(i,j)+Hx(i,1,j);
           mult=mult+1;
       end
       if(Hx(i,2,j)~=0)
           Hx_grid(i,j)=Hx_grid(i,j)+Hx(i,2,j);
           mult=mult+1;
       end
       if(mult>0)
           Hx_grid(i,j)=Hx_grid(i,j)/mult;
       else
           Hx_grid(i,j)=0;
       end
       
       mult=0;
       if(Hy(i,1,j)~=0)
           Hy_grid(i,j)=Hy_grid(i,j)+Hy(i,1,j);
           mult=mult+1;
       end
       if(Hy(i,2,j)~=0)
           Hy_grid(i,j)=Hy_grid(i,j)+Hy(i,2,j);
           mult=mult+1;
       end
       if(mult>0)
           Hy_grid(i,j)=Hy_grid(i,j)/mult;
       else
           Hy_grid(i,j)=0;
       end

       Hmod_grid(i,j)=sqrt( Hx_grid(i,j)^2+Hy_grid(i,j)^2)*1000;
   end
end

