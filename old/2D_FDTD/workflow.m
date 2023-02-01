% workflow
clc;
grid_definition;

fig=figure('Position',[100 100 500 270]);

ind=1;
for p=1:1:200
    Niter=p; % number of time steps
    hyperbolic_2d_fdtd;
    solution_plot;
    axis off;
    saveas(fig,['Step',num2str(ind)],'bmp');
    ind=ind+1;
end
