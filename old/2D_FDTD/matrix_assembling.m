% matrix assemling

for i=1:Ng
    for j=1:Ng
        A(i,j)=0.;
    end
end

for i=1:Ng % loop over nodes
    if(type_bc(i)==0) %regular node
       il=neighbors(i,1); %left
       ir=neighbors(i,2); %right
       ib=neighbors(i,3); %bottom
       it=neighbors(i,4); %top
       % matrix
       A(i,i)=6.;
       A(i,il)=-1.;
       A(i,ir)=-1.;
       A(i,ib)=-1.;
       A(i,it)=-1.;
    end
    if(type_bc(i)==1) %dirichlet node
       A(i,i)=1.;
       b(i)=value_bc(i);
    end
    if(type_bc(i)==2) %neumann node
       il=neighbors(i,1); %left
       ir=neighbors(i,2); %right
       ib=neighbors(i,3); %bottom
       it=neighbors(i,4); %top
       % matrix
       A(i,i)=6.;

       if(il==0)
          A(i,i)=A(i,i)-1.;
       else
          A(i,il)=-1.;
       end
       
       if(ir==0)
          A(i,i)=A(i,i)-1.;
       else
          A(i,ir)=-1.;
       end
       
       if(ib==0)
          A(i,i)=A(i,i)-1.;
        else
          A(i,ib)=-1.;
       end
       
       if(it==0)
          A(i,i)=A(i,i)-1.;
       else
          A(i,it)=-1.;
       end
       
    end
 
end

