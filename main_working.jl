const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics

@parallel_indices (iy) function insert_wave!(p,cosine_wave,omega,t)
        p[1,iy] = sin(omega*t).*cosine_wave[iy]

    return
end

@parallel_indices (ix,iy) function compute_u!(u,p,dt_over_rho_x_dx)
    u[ix,iy] = u[ix,iy] - dt_over_rho_x_dx * ( p[ix,iy] - p[ix-1,iy] )

    return
end

@parallel_indices (ix,iy) function compute_v!(v,p,dt_over_rho_x_dy)
    v[ix,iy] = v[ix,iy] - dt_over_rho_x_dy * ( p[ix,iy] - p[ix,iy-1] )

    return
end


# @parallel function compute_p!(p,u,v,k,dt,dx,dy)
#     @all(p) = @all(p) - dt*k*(@d_xa(u)/dx+@d_ya(v)/dy)
    
#     return
# end

# @parallel function compute_u!(u,p,density,dt,dx)
#     @inn(u) = @inn(u) + dt/(density*dx)*@d_xi(p)
#     return
# end

# @parallel function compute_v!(v,p,density,dt,dy)
#     @inn(v) = @inn(v) + dt/(density*dy)*@d_yi(p)
#     return
# end

@parallel_indices (iy) function exchange_ghosts!(p_gas,p_liquid,lower_y_bound,upper_y_bound)
    #exchange ghost cells
    # p_liquid[1,lower_y_bound:upper_y_bound] = p_gas[end-1,:]
    # p_gas[end,:] = p_liquid[2,lower_y_bound:upper_y_bound]
    p_liquid[1,lower_y_bound+iy] = p_gas[end-1,iy]
    p_gas[end,iy] = p_liquid[2,lower_y_bound+iy]

    return
end





##################################################
@views function acoustic2D()
    #Physics
    lx_gas, ly_gas          = 0.02, 0.02  # domain extends of gas [m]
    lx_liquid, ly_liquid    = 0.02, 0.04  # domain extends of liquid [m]
    k_gas                   = 149470      # bulk modulus of gas (set such that speed of sound is 340m/s)
    k_liquid                = 1e7      # bulk modulus of liquid (set such that speed of sound is 1500m/s)

    ρ_gas                   = 1.293       # density of gas [kg/m^3]
    ρ_liquid                = 45         # density of liquid [kg/m^3]
    t                       = 0.0         # physical time [s]
    pi                      = 3.14159265  # pi
    frequency               = 100000    # frequency of the wave [Hz]

    #Derived physics
    omega = 2*pi*frequency          #angular frequency of the wave
    c_gas_const = sqrt(k_gas/ρ_gas)           # speed of sound in gas
    c_liquid_const = sqrt(k_liquid/ρ_liquid)     # speed of sound in liquid


    # Numerics
    nx_gas, ny_gas    = 510, 510  # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx_liquid, ny_liquid = 510, 2*ny_gas-1
    nt        = 10000       # number of timesteps
    nout      = 100        # plotting frequency
    
    # Derived numerics
    dx_gas, dy_gas          = lx_gas/(nx_gas-1), ly_gas/(ny_gas-1)              # cell sizes in gas
    dx_liquid, dy_liquid    = lx_liquid/(nx_liquid-1), ly_liquid/(ny_liquid-1)  # cell sizes in liquid
    lower_y_bound = floor(Int,(ny_liquid-ny_gas)/2)
    upper_y_bound = lower_y_bound + ny_gas - 1

    # Array allocations
    p_gas               = @zeros(nx_gas,ny_gas)
    p_liquid            = @zeros(nx_liquid,ny_liquid)
    u_gas               = @zeros(nx_gas+1,ny_gas)
    u_liquid            = @zeros(nx_liquid+1,ny_liquid)
    v_gas               = @zeros(nx_gas,ny_gas+1)
    v_liquid            = @zeros(nx_liquid,ny_liquid+1)
    p_liquid_old        = @zeros(nx_liquid,ny_liquid)
    cosine_wave         = @zeros(ny_gas)


    # Initial conditions
    dt        = min(dx_gas,dy_gas)/sqrt(k_gas/ρ_gas)/4.1#adjust for both k_gas and k_liquid (probably min)
    CFL = c_liquid_const*dt/dx_liquid
    X, Y      = -(lx_gas+lx_liquid)/2:lx_gas/(nx_gas-1.5):(lx_gas+lx_liquid)/2, -ly_liquid/2:dy_liquid:ly_liquid/2
    X_gas = 0:dy_gas:ly_gas
    X_gas = X_gas.*2*pi/ly_gas
    X_gas = cos.(X_gas)
    for i = 1:ny_gas
        cosine_wave[i] = X_gas[i]
    end

    dt_over_rho_x_dx_gas = dt / (ρ_gas * dx_gas)
    dt_over_rho_x_dy_gas = dt / (ρ_gas * dy_gas)
    dt_over_rho_x_dx_liquid = dt / (ρ_liquid * dx_liquid)
    dt_over_rho_x_dy_liquid = dt / (ρ_liquid * dy_liquid)
    kappa_x_dt_over_dx_gas = k_gas * dt / dx_gas
    kappa_x_dt_over_dx_liquid = k_liquid * dt / dx_liquid

    # Prepare visualization
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    p_gas_plot  = @zeros(nx_gas,ny_liquid)
    p_gas_plot .= 0
    println("Animation directory: $(anim.dir)")

    # Time loop
    for iter = 1:nt
        # Inject Wave
        @parallel (1:size(p_gas, 2)) insert_wave!(p_gas,cosine_wave,omega,t)

        @parallel (1:size(p_gas, 2)) exchange_ghosts!(p_gas,p_liquid,lower_y_bound,upper_y_bound)

        # Save pressure from time step "n" for BC computation
        p_liquid_old .= p_liquid

        # Update u
        # u_gas[2:nx_gas,:] = u_gas[2:nx_gas,:] - dt_over_rho_x_dx_gas * ( p_gas[2:nx_gas,:] - p_gas[1:nx_gas-1,:] )
        # u_liquid[2:nx_liquid,:] = u_liquid[2:nx_liquid,:] - dt_over_rho_x_dx_liquid * ( p_liquid[2:nx_liquid,:] - p_liquid[1:nx_liquid-1,:] )
        @parallel (2:size(u_gas, 1)-1,1:size(u_gas, 2)) compute_u!(u_gas,p_gas,dt_over_rho_x_dx_gas)
        @parallel (2:size(u_liquid, 1)-1,1:size(u_liquid, 2)) compute_u!(u_liquid,p_liquid,dt_over_rho_x_dx_liquid)
        # Update v
        # v_gas[:,2:ny_gas] = v_gas[:,2:ny_gas] - dt_over_rho_x_dy_gas * ( p_gas[:,2:ny_gas] - p_gas[:,1:ny_gas-1] )
        # v_liquid[:,2:ny_liquid] = v_liquid[:,2:ny_liquid] - dt_over_rho_x_dy_liquid * ( p_liquid[:,2:ny_liquid] - p_liquid[:,1:ny_liquid-1] )
        @parallel (1:size(v_gas, 1),2:size(v_gas, 2)-1) compute_v!(v_gas,p_gas,dt_over_rho_x_dy_gas)
        @parallel (1:size(v_liquid, 1),2:size(v_liquid, 2)-1) compute_v!(v_liquid,p_liquid,dt_over_rho_x_dy_liquid)
        # Update p
        p_gas = p_gas - kappa_x_dt_over_dx_gas * ( ( u_gas[2:nx_gas+1,:] - u_gas[1:nx_gas,:] ) + ( v_gas[:,2:ny_gas+1]  - v_gas[:,1:ny_gas]  ))
        p_liquid = p_liquid - kappa_x_dt_over_dx_liquid * ( ( u_liquid[2:nx_liquid+1,:] - u_liquid[1:nx_liquid,:] ) + ( v_liquid[:,2:ny_liquid+1]  - v_liquid[:,1:ny_liquid]  ))

        # Absorption BC begin
        # East
        p_liquid[end,2:end-1] =  p_liquid_old[end-1,2:end-1] + (CFL-1)/(CFL+1)*(p_liquid[end-1,2:end-1] - p_liquid_old[end,2:end-1])
        # South
        p_liquid[2:end-1,1] = p_liquid_old[2:end-1,2] + (CFL-1)/(CFL+1)*(p_liquid[2:end-1,2] - p_liquid_old[2:end-1,1])
        # North
        p_liquid[2:end-1,end] = p_liquid_old[2:end-1,end-1] + (CFL-1)/(CFL+1)*(p_liquid[2:end-1,end-1] - p_liquid_old[2:end-1,end])
        # West
        p_liquid[1,2:lower_y_bound-1] = p_liquid_old[2,2:lower_y_bound-1] + (CFL-1)/(CFL+1)*(p_liquid[2,2:lower_y_bound-1] - p_liquid_old[1,2:lower_y_bound-1])        
        p_liquid[1,upper_y_bound+1:end-1] = p_liquid_old[2,upper_y_bound+1:end-1] + (CFL-1)/(CFL+1)*(p_liquid[2,upper_y_bound+1:end-1] - p_liquid_old[1,upper_y_bound+1:end-1])
        # Absorption BC end
        
        t += dt
        # Visualization
        if mod(iter,nout)==0
            println("iter = $iter, t = $t")
            p_gas_plot[:,lower_y_bound:upper_y_bound] = p_gas
            heatmap(X, Y, Array([p_gas_plot[1:end-1,:]; p_liquid[2:end,:]])',aspect_ratio=1, 
            xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure",dpi=300); frame(anim)
        end

    end

    # Save animation
    gif(anim, "acoustic2D.gif", fps = 10)
    
    return
end

acoustic2D()