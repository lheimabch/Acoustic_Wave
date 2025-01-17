const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics
# default(size=(1200,800),framestyle=:box,label=false,grid=false,margin=35mm,lw=6,labelfontsize=20,tickfontsize=20,titlefontsize=24)
#need to add keyword parallel to all functions to enable 
@parallel function compute_p!(p, v, f, dt)
    @all(p) = @all(p) + @all(v)*dt + 0.5*@all(f)*dt*dt
    # for it = 1:size(p,1)
    #     for jt = 1:size(p,2)
    #         p[it,jt] = p[it,jt] + v[it,jt]*dt + 0.5*f[it,jt]*dt*dt
    #     end
    # end

    return
end

@parallel function compute_v!(v, f, f_new, dt)
    @all(v) = @all(v) + 0.5*(@all(f) + @all(f_new))*dt
    # for it = 1:size(v,1)
    #     for jt = 1:size(v,2)
    #         v[it,jt] = v[it,jt] + 0.5*(f[it,jt] + f_new[it,jt])*dt
    #     end
    # end

    return
end

@parallel function compute_f!(c,p,f,dx,dy)
    @inn(f) = @inn(c)*(@d2_xi(p)/(dx*dx) + @d2_yi(p)/(dy*dy))
    # for it = 2:size(f,1)-1
    #     for jt = 2:size(f,2)-1
    #         f[it,jt] = c[it,jt]*((p[it+1,jt] - 2*p[it,jt] + p[it-1,jt])/(dx*dx) + (p[it,jt+1] - 2*p[it,jt] + p[it,jt-1])/(dy*dy))
    #     end
    # end
    
    return
end

@parallel_indices (iy) function insert_wave!(p,cosine_wave,omega,t)
        p[end,iy] = sin(omega*t).*cosine_wave[iy]


    return
end

@parallel_indices (iy) function exchange_ghosts!(p_gas,p_liquid,lower_y_bound,upper_y_bound)
    #exchange ghost cells
    # p_liquid[1,lower_y_bound:upper_y_bound] = p_gas[end-1,:]
    # p_gas[end,:] = p_liquid[2,lower_y_bound:upper_y_bound]
    p_liquid[1,lower_y_bound+iy] = p_gas[end-1,iy]
    p_gas[end,iy] = p_liquid[2,lower_y_bound+iy]

    return
end

@parallel_indices (iy) function absorption_boundary_x_direction!(p_liquid,p_liquid_n,c_liquid,dx,dt)
    #east boundary
    # p_liquid[end,iy] = - p_liquid_n_minus_1[end-1,iy] + (c_liquid*dt-dx)/(c_liquid*dt+dx)*(p_liquid_n_minus_1[end,iy] + p_liquid[end-1,iy]) + 2*dx/(c_liquid*dt+dx)*(p_liquid_n[end,iy]+p_liquid_n[end-1,iy])
    # v_liquid[end,iy] =  c_liquid*(p_liquid[end-1,iy]-p_liquid[end,iy])/dx
    p_liquid[1,iy] = p_liquid_n[2,iy] + (c_liquid*dt-dx)/(c_liquid*dt+dx)*(p_liquid[2,iy]-p_liquid_n[1,iy])

    return
end

@parallel_indices (iy) function absorption_boundary_v_direction!(v_liquid,v_liquid_n,c_liquid,dx,dt)
v_liquid[1,iy] = v_liquid_n[2,iy] + (c_liquid*dt-dx)/(c_liquid*dt+dx)*(v_liquid[2,iy]-v_liquid_n[1,iy])
    return
end
@parallel_indices (ix) function absorption_boundary_y_N_direction!(p_liquid,c_liquid,dy,dt)
    #east boundary
    p_liquid[ix,end] = p_liquid[ix,end-1] + c_liquid*dt/dy.*(p_liquid[ix,end-1]-p_liquid[ix,end])

    return
end

@parallel_indices (ix) function absorption_boundary_y_S_direction!(p_liquid,c_liquid,dx,dt)
    #east boundary
    p_liquid[ix,1] = p_liquid[ix,1] + c_liquid*dt/dx.*(p_liquid[ix,2]-p_liquid[ix,1])

    return
end


##################################################
@views function acoustic2D()
    #Physics
    lx_gas, ly_gas          = 0.02, 0.02  # domain extends of gas [m]
    lx_liquid, ly_liquid    = 0.02, 0.04  # domain extends of liquid [m]
    k_gas                   = 149470      # bulk modulus of gas (set such that speed of sound is 340m/s)
    k_liquid                = 2.2e9       # bulk modulus of liquid (set such that speed of sound is 1500m/s)

    ρ_gas                   = 1.293       # density of gas [kg/m^3]
    ρ_liquid                = 997         # density of liquid [kg/m^3]
    t                       = 0.0         # physical time [s]
    pi                      = 3.14159265  # pi
    frequency               = 10000    # frequency of the wave [Hz]

    #Derived physics
    omega = 2*pi*frequency          #angular frequency of the wave
    c_gas_const = sqrt(k_gas/ρ_gas)           # speed of sound in gas
    c_liquid_const = sqrt(k_liquid/ρ_liquid)     # speed of sound in liquid


    # Numerics
    nx_gas, ny_gas    = 510, 510  # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx_liquid, ny_liquid = 510, 2*ny_gas-1
    nt        = 80000       # number of timesteps
    nout      = 1000        # plotting frequency
    
    # Derived numerics
    dx_gas, dy_gas          = lx_gas/(nx_gas-1), ly_gas/(ny_gas-1)              # cell sizes in gas
    dx_liquid, dy_liquid    = lx_liquid/(nx_liquid-1), ly_liquid/(ny_liquid-1)  # cell sizes in liquid
    lower_y_bound = floor(Int,(ny_liquid-ny_gas)/2)
    upper_y_bound = lower_y_bound + ny_gas - 1

    # Array allocations
    p_gas               = @zeros(nx_gas,ny_gas)
    p_gas_plot          = @zeros(nx_gas,ny_liquid)
    p_liquid            = @zeros(nx_liquid,ny_liquid)
    u_gas               = @zeros(nx_gas,ny_gas)
    v_gas               = @zeros(nx_gas,ny_gas)
    u_liquid            = @zeros(nx_liquid,ny_liquid)
    v_liquid            = @zeros(nx_liquid,ny_liquid)



    # Initial conditions
    dt        = min(dx_gas,dy_gas)/sqrt(k_gas/ρ_gas)/4.1#adjust for both k_gas and k_liquid (probably min)
    c_gas    .= c_gas_const
    c_liquid .= c_liquid_const
    X, Y      = -(lx_gas+lx_liquid)/2:lx_gas/(nx_gas-1.5):(lx_gas+lx_liquid)/2, -ly_liquid/2:dy_liquid:ly_liquid/2
    X_gas = 0:dy_gas:ly_gas
    X_gas = X_gas.*2*pi/ly_gas
    X_gas = cos.(X_gas)
    for i = 1:ny_gas
        cosine_wave[i] = X_gas[i]
    end



    # Prepare visualization
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    p_gas_plot .= 0
    println("Animation directory: $(anim.dir)")

    # Time loop
    for iter = 1:nt
        if(t<=0.5/frequency)
            # @parallel (1:size(p_gas, 2)) insert_wave!(p_gas,cosine_wave,omega,t)
            p_liquid[end,:] .= 1
        end
        if(t>0.5/frequency)
            p_liquid[end,:] .= 0
        end
        if (iter==1)
            @parallel compute_f!(c_gas,p_gas,f_current_gas,dx_gas,dy_gas)
            @parallel compute_f!(c_liquid,p_liquid,f_current_liquid,dx_liquid,dy_liquid)
        else
            f_current_gas .= f_new_gas
            f_current_liquid .= f_new_liquid
        end
        # @parallel (1:size(p_gas, 2)) exchange_ghosts!(p_gas,p_liquid,lower_y_bound,upper_y_bound)
        p_liquid_n .= p_liquid
        @parallel compute_p!(p_gas, v_gas, f_current_gas, dt)
        @parallel compute_p!(p_liquid, v_liquid, f_current_liquid, dt)

        @parallel (1:size(p_liquid, 2)) absorption_boundary_x_direction!(p_liquid,p_liquid_n,c_liquid_const,dx_liquid,dt)

        # @parallel (1:size(p_liquid, 1)) absorption_boundary_y_N_direction!(p_liquid,c_liquid_const,dy_liquid,dt)
        # @parallel (1:size(p_liquid, 1)) absorption_boundary_y_S_direction!(p_liquid,c_liquid_const,dy_liquid,dt)

        v_liquid_n .= v_liquid
        @parallel compute_f!(c_gas,p_gas,f_new_gas,dx_gas,dy_gas)
        @parallel compute_f!(c_liquid,p_liquid,f_new_liquid,dx_liquid,dy_liquid)
        @parallel compute_v!(v_gas, f_current_gas, f_new_gas, dt)
        @parallel compute_v!(v_liquid, f_current_liquid, f_new_liquid, dt)
        @parallel (1:size(v_liquid, 2)) absorption_boundary_v_direction!(v_liquid,v_liquid_n,c_liquid_const,dx_liquid,dt)

        t += dt

        if mod(iter,nout)==0
            println("iter = $iter, t = $t")

            p_gas_plot[:,lower_y_bound:upper_y_bound] = p_gas
            heatmap(X, Y, Array([p_gas_plot[1:end-1,:]; p_liquid[2:end,:]])',aspect_ratio=1, 
            xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure",dpi=300); frame(anim)

        end

    end

    # Save animation
    gif(anim, "acoustic2D.gif", fps = 15)
    
    return
end

acoustic2D()