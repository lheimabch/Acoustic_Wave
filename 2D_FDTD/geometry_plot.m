% plot geometry

for i=1:Ng
   if(type_bc(i)>0) % boundary
       n_lines=0;
       for j=1:4 % loop over neighbors
           ng=neighbors(i,j);
           if(ng>0)
               if(type_bc(ng)>0) % neighbor is also boundary
                   line(x(i),y(i),x(ng),y(ng),'k',2);
                   n_lines=n_lines+1;
               end
           end
       end   
       if(n_lines==0) % additional testing is needed
           for j=1:4
               ng=neighbors(i,j);
               if(ng>0)
                    for p=1:4
                       ng1=neighbors(ng,p);
                       if(type_bc(ng1)>0) % indirect boundary
                            line(x(i),y(i),x(ng1),y(ng1),'k',2);
                       end
                    end
              end
           end
       end
   end
end
