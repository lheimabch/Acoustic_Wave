% plot magnetic field

fig=figure('Position',[100 100 500 270]);

ind=1;

for p=3:1:200
    % Niter=p; % number of time steps
    % hyperbolic_2d_fdtd;

    step=p;
    
    for i=1:Ng
        k=int8(x(i)/h);
        l=int8(y(i)/h);
        % P(l,k)=E(i,step);
        P(l,k)=Hmod_grid(i,step);
    end

    time=step*delta_t;

    solution_plot;
    axis off;
    saveas(fig,['Step',num2str(ind)],'bmp');
    ind=ind+1;
end
