# Solve -u'' + V u + α u^3 = f

using FFTW
using NLsolve
using KrylovKit

N = 100
V(x) = 2 + sin(x)
f(x) = cos(x)
const α = 1.0

# frequencies, avoiding aliasing: eg if N=5, this is 0, 1, 2, -2, 1
ω(i, N) = (i <= div(N,2)+1) ? (i-1) : -(N-i+1)

const x = range(0, 2π, length=N+1)[1:end-1]
const ωs = ω.(1:N, N)
const Vx = V.(x)
const fx = f.(x)

lap(u) = real(ifft(ωs.^2 .* fft(u))) # computes u''
precond(u) = real(ifft(fft(u) ./ (1 .+ ωs .^ 2)))

residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u

# we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
function fp_frozen(u_frozen)
    linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)[1]
end

u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson, show_trace=true).zero
maximum(abs, residual(u_sol))

## directly trying to solve without splitting doesn't work out numerically
# nlsolve(u -> residual(u), zeros(N), method = :anderson, show_trace=true).zero

#===#
# TODO: find e.g. d(sum(u*)) / dVx via AD

#===#
# Approach 1: Direct black-box AD with Zygote (unrolling solvers in time)

using Zygote
using LinearMaps
using IterativeSolvers
using BenchmarkTools

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)[1]
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson, show_trace=true).zero
        sum(u_sol)
    end
    @time blackbox(Vx)
    Zygote.gradient(blackbox, Vx) # ERROR: LoadError: Mutating arrays is not supported
end

#===#
# Approach 2: Add the general adjoint of nlsolve

Zygote.@adjoint nlsolve(f, x0; kwargs...) =
    let result = nlsolve(f, x0; kwargs...)
        result, function(vresult)
            dx = vresult[].zero
            x = result.zero
            _, back_x = Zygote.pullback(f, x)

            JT(df) = back_x(df)[1]
            # solve JT*df = -dx
            L = LinearMap(JT, length(x0))
            df = gmres(L,-dx)

            _, back_f = Zygote.pullback(f -> f(x), f)
            return (back_f(df)[1], nothing, nothing)
        end
    end
Zygote.refresh()

#========#
# Try with NLsolve only (since we already defined its adjoint above)

using FiniteDiff
using BenchmarkTools

Float64(z::Complex) = Float64(real(z)) # TODO Hack...

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            nlsolve(u -> precond(lin_op(u, u_frozen)) - precond(fx), zeros(N), method = :anderson).zero
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson).zero
        sum(u_sol)
    end
    @time blackbox(Vx) # 0.605172 seconds (1.47 M allocations: 105.368 MiB, 10.58% gc time, 98.51% compilation time)
    @time g1 = real(Zygote.gradient(blackbox, Vx)[1]) # 10.130100 seconds (16.95 M allocations: 982.691 MiB, 5.67% gc time, 99.35% compilation time)
    # g2 = ForwardDiff.gradient(blackbox, Vx)
    @time g3 = FiniteDiff.finite_difference_gradient(blackbox, Vx) # 3.132538 seconds (4.94 M allocations: 846.075 MiB, 10.13% gc time, 14.61% compilation time)
    @show g1[1:5]
    @show g3[1:5]
    @show sum(abs, g1 - g3) / length(g1) # 0.014476332143572825
    @btime $blackbox($x) # 21.492 ms (41011 allocations: 8.33 MiB)
    @btime real(Zygote.gradient($blackbox, $Vx)[1]) # 28.739 ms (116532 allocations: 15.70 MiB)
    nothing
end # looks promising


# TODO next step: implement an adjoint rule for linearsolve or gmres

using LinearAlgebra
using ChainRulesCore

function ChainRulesCore.rrule(::typeof(linsolve), A::Function, B; kwargs...)
    # loosely https://github.com/JuliaDiff/ChainRules.jl/blob/67c106fe32d797d18135a6d699636a8bfca41760/src/rulesets/Base/arraymath.jl#L125
    # additional assumption: A is symmetric i.e. A(x) == A'(x), to avoid the transpose of a Function.
    # Y = A \ B
    println("👋linsolve rrule") 
    Y = linsolve(A, B; kwargs...)
    function backslash_pullback(Ȳ)
        println("👈linsolve pullback") 
        ∂A = @thunk begin
            B̄ = linsolve(A, Ȳ[1]; kwargs...)[1]
            a = (-B̄ * Y[1]')
            b = linsolve(A, B̄ * (B - A(Y[1]))')[1]'
            c = linsolve(A, Y[1])[1]*(Ȳ[1]' - A(B̄)')
            d = a + b + c
            Ā = x -> d*x
            Ā
        end
        ∂B = @thunk linsolve(A, Ȳ[1]; kwargs...)[1]
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback
end
Zygote.refresh()

## check if the linsolve rrule works on a small linear system
a = let
    x = rand(10, 100)
    x * x' + I
end
b = rand(10)

out, back = rrule(linsolve, x->a*x, b)
df = unthunk(back(out)[2])
df(out[1])

out2, back2 = Zygote.pullback(A -> A \ b, a)
back2(out2)
out2

out[1] ≈ out2
back2(out2)
back2(out[1])[1]*out[1]
back2(out[1])[1]*out[1] ≈ df(out[1])

# ^seems to work
# TODO now try on the ode (use linsolve inside fp_frozen again)

let 
    blackbox = Vx -> begin
        residual(u) = -lap(u) .+ @. Vx * u + α * u^3 - fx
        lin_op(u, u_frozen) = -lap(u) .+ @. Vx * u + α * u_frozen^2 * u
        # we solve the fixed-point u_frozen = g(u_frozen) where g(u_frozen) is the solution to the linear equation -u'' + Vu + α ufrozen^2 u = f
        function fp_frozen(u_frozen)
            linsolve(u -> precond(lin_op(u, u_frozen)), precond(fx), tol=1e-14)[1]
        end
        u_sol = nlsolve(u -> fp_frozen(u) - u, zeros(N), method = :anderson).zero
        sum(u_sol)
    end
    @time blackbox(Vx)
    @time g1 = real(Zygote.gradient(blackbox, Vx)[1]) # LoadError: Need an adjoint for constructor var"#88#94"{Vector{Float64}, var"#lin_op#92"{Vector{Float64}}}. Gradient is of type var"#75#79"{Matrix{Float64}}
    # g2 = ForwardDiff.gradient(blackbox, Vx)
    # @time g3 = FiniteDiff.finite_difference_gradient(blackbox, Vx)
    # @show g1[1:5]
    # @show g3[1:5]
    # @show sum(abs, g1 - g3) / length(g1)
    # @btime $blackbox($x)
    # @btime real(Zygote.gradient($blackbox, $Vx)[1])
    # nothing
end

