const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics

#need to add keyword parallel to all functions to enable 
function compute_p!(p, v, f, dt)
    # @all(p) = @all(p) + @all(v)*dt + 0.5*@all(f)*dt*dt
    for it = 2:size(p,1)-1
        for jt = 2:size(p,2)-1
            p[it,jt] = p[it,jt] + v[it,jt]*dt + 0.5*f[it,jt]*dt*dt
        end
    end

    return
end

function compute_v!(v, f, f_new, dt)
    # @all(v) = @all(v) + 0.5*(@all(f) + @all(f_new))*dt
    for it = 2:size(v,1)-1
        for jt = 2:size(v,2)-1
            v[it,jt] = v[it,jt] + 0.5*(f[it,jt] + f_new[it,jt])*dt
        end
    end

    return
end

function compute_f!(c,p,f,dx,dy)
    # @inn(f) = @inn(c)*(@d2_xi(p)/(dx*dx) + @d2_yi(p)/(dy*dy))
    for it = 2:size(f,1)-1
        for jt = 2:size(f,2)-1
            f[it,jt] = c[it,jt]*( (p[it+1,jt] - 2*p[it,jt] + p[it-1,jt])/(dx*dx) + (p[it,jt+1] - 2*p[it,jt] + p[it,jt-1])/(dy*dy) )
        end
    end
    
    return
end

function insert_wave!(p,frequency,t,dx,L,pi)
    for jt = 1:size(p,2)
        p[1,jt] = sin(frequency*t)*cos(pi*jt*dx/L)
    end


    return
end

function exchange_ghosts!(p_gas,p_liquid)

    return
end


##################################################
@views function acoustic2D()
    #Physics
    lx_gas, ly_gas          = 20.0, 20.0  # domain extends of gas
    lx_liquid, ly_liquid    = 20.0, 20.0  # domain extends of liquid
    k                       = 1.0         # bulk modulus
    ρ_gas                   = 1.0         # density of gas
    ρ_liquid                = 1.0         # density of liquid
    t                       = 0.0         # physical time
    frequency               = 1000        # frequency of the wave
    pi                      = 3.14159265  # pi

    #Derived physics
    c_gas_cosnt = sqrt(k/ρ_gas)           # speed of sound in gas
    c_liquid_const = sqrt(k/ρ_liquid)        # speed of sound in liquid

    # Numerics
    nx_gas, ny_gas    = 255, 255    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx_liquid, ny_liquid = 255, 255
    nt        = 1000        # number of timesteps
    nout      = 10          # plotting frequency

    # Derived numerics
    dx_gas, dy_gas          = lx_gas/(nx_gas-1), ly_gas/(ny_gas-1)              # cell sizes in gas
    dx_liquid, dy_liquid    = lx_liquid/(nx_liquid-1), ly_liquid/(ny_liquid-1)  # cell sizes in liquid

    # Array allocations
    p_gas         = @zeros(nx_gas,ny_gas)
    p_liquid      = @zeros(nx_liquid,ny_liquid)
    c_gas         = @zeros(nx_gas,ny_gas)
    c_liquid      = @zeros(nx_liquid,ny_liquid)
    v_gas         = @zeros(nx_gas,ny_gas)
    v_liquid      = @zeros(nx_liquid,ny_liquid)
    f_new         = @zeros(nx_gas,ny_gas)
    f_current     = @zeros(nx_gas,ny_gas)

    # Initial conditions
    dt        = min(dx_gas,dy_gas)/sqrt(k/ρ_gas)/4.1
    p_gas.= Data.Array([exp(-((ix-1)*dx_gas-0.5*lx_gas)^2 -((iy-1)*dy_gas-0.5*ly_gas)^2) for ix=1:size(p_gas,1), iy=1:size(p_gas,2)])
    c_gas .= c_gas_cosnt
    compute_f!(c_gas,p_gas,f_current,dx_gas,dy_gas)
    X, Y      = -lx_gas/2:dx_gas:lx_gas/2, -ly_gas/2:dy_gas:ly_gas/2

    # Prepare visualization
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")

    # Time loop
    for iter = 1:nt
        if (iter!=1)
            f_current = f_new
        end
        insert_wave!(p_gas,frequency,t,dx_gas,lx_gas,pi)
        compute_p!(p_gas, v_gas, f_current, dt)
        compute_f!(c_gas,p_gas,f_new,dx_gas,dy_gas)
        compute_v!(v_gas, f_current, f_new, dt)
        t += dt

        if mod(iter,nout)==0
            heatmap(X, Y, Array(p_gas)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure"); frame(anim)
        end

    end

    # Save animation
    gif(anim, "acoustic2D.gif", fps = 15)
    
    return
end

acoustic2D()