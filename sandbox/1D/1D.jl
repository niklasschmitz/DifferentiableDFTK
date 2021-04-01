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

#===#
# TODO: find e.g. d(sum(u*)) / dVx via AD

## directly trying to solve without splitting doesn't work out numerically
# nlsolve(u -> residual(u), zeros(N), method = :anderson, show_trace=true).zero

using Zygote
using LinearMaps
using IterativeSolvers

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
    Zygote.gradient(blackbox, Vx)
end

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

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(linsolve), A::Function, B; kwargs...)
    # loosely https://github.com/JuliaDiff/ChainRules.jl/blob/67c106fe32d797d18135a6d699636a8bfca41760/src/rulesets/Base/arraymath.jl#L125
    # additional assumption: A is symmetric i.e. A(x) == A'(x), to avoid the transpose of a Function.
    # Y = A \ B
    Y = linsolve(A, B; kwargs...)
    function backslash_pullback(Ȳ)
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

using LinearAlgebra

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


#========#
# Try with NLsolve only (since we already defined its adjoint above)

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
    @time blackbox(Vx)
    @time g1 = real(Zygote.gradient(blackbox, Vx)[1])
    # g2 = ForwardDiff.gradient(blackbox, Vx)
    @time g3 = FiniteDiff.finite_difference_gradient(blackbox, Vx)
    @show g1[1:5]
    @show g3[1:5]
    @show sum(abs, g1 - g3) / length(g1)
end

# TODO next step: implement an adjoint rule for linearsolve or gmres
