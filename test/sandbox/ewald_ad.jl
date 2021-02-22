include("ewald.jl")

using ForwardDiff
using ReverseDiff
using FiniteDiff

using LinearAlgebra
using Test
using BenchmarkTools


#= helpers =#

function compute_forces_manual(positions, lattice, charges)
    forces_manual = zeros(Vec3{Float64}, length(positions))
    γ1 = energy_ewald(lattice, charges, positions, forces=forces_manual)
    forces_manual = reduce(hcat, forces_manual)
    return forces_manual
end

function compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ForwardDiff)
    _positions = reduce(hcat, positions)
    if autodiff_backend == :FiniteDiff
        gradient = FiniteDiff.finite_difference_gradient
    else
        gradient = eval(autodiff_backend).gradient
    end 
    forces = gradient(_positions -> begin
            positions = collect(eachcol(_positions))
            -energy_ewald(lattice, charges, positions)
        end,
        _positions
    )
    return forces
end


#==========#
# Forces   #
#==========#

@testset "Forces" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    charges = [14, 14]

    forces = zeros(Vec3{Float64}, 2)
    γ1 = energy_ewald(lattice, charges, positions, forces=forces)

    # Compare forces to finite differences
    disp = [rand(3)/20, rand(3)/20]
    ε = 1e-8
    γ2 = energy_ewald(lattice, charges, positions .+ ε .* disp)
    @test (γ2-γ1)/ε ≈ -dot(disp, forces) atol=abs(γ1*1e-6)
end

begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    charges = [14, 14]

    # compute forces manually
    forces_manual = compute_forces_manual(positions, lattice, charges)

    # compute forces via ForwardDiff
    forces_forwarddiff = compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ForwardDiff)

    display(forces_manual)
    display(forces_forwarddiff)

    @test forces_manual ≈ forces_forwarddiff atol=1e-6
end

# Ewald needed for ForwardDiff: No annotation of eltype T to allow overloading.

@time compute_forces_manual(positions, lattice, charges)
@time compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:FiniteDiff)
@time compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ForwardDiff)
@time compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ReverseDiff)
# compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:Zygote) # Error


#=====================#
# Compile ReverseDiff # https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl
#=====================#
# While not strictly required, ReverseDiff recommends 
# compiling an explicit tape that pre-records operations

@time let 
    _positions = reduce(hcat, positions)
    compiled_energy_tape = ReverseDiff.compile(
        ReverseDiff.GradientTape(
            _positions -> begin
                positions = collect(eachcol(_positions))
                -energy_ewald(lattice, charges, positions)
            end,
            rand(3,2)
        )
    )
    res = similar(_positions)
    ReverseDiff.gradient!(res, compiled_energy_tape, _positions)
end


#=====================================#
# Forward vs Reverse on larger system # TODO: ask if system is still physically meaningful
#=====================================#

function benchmark_ewald(num_atoms=20)
    lattice = [0.0  5.131570667152971 5.131570667152971;
    5.131570667152971 0.0 5.131570667152971;
    5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20 for _ in 1:num_atoms]
    charges = [14 for _ in 1:num_atoms]

    # advanced setup for ReverseDiff
    compiled_energy_tape = ReverseDiff.compile(
        ReverseDiff.GradientTape(
            _positions -> begin
                pos = collect(eachcol(_positions))
                -energy_ewald(lattice, charges, pos)
            end,
            rand(3, num_atoms)
        )
    )    

    forces_manual = compute_forces_manual(positions, lattice, charges)
    forces_forwarddiff = compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ForwardDiff)

    _positions = reduce(hcat, positions)
    forces_reversediff = similar(_positions)
    ReverseDiff.gradient!(forces_reversediff, compiled_energy_tape, _positions)
    
    # check consistency of forces
    @test forces_manual ≈ forces_forwarddiff atol=1e-6
    @test forces_manual ≈ forces_reversediff atol=1e-6

    println("Timings:")
    println("Energy without forces")
    @btime energy_ewald(lattice, charges, positions)
    println("Manual forces")
    @btime compute_forces_manual(positions, lattice, charges)
    println("FiniteDiff forces")
    @btime compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:FiniteDiff)
    println("ForwardDiff forces")
    @btime compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ForwardDiff)
    println("ReverseDiff forces")
    @btime compute_forces_autodiff(positions, lattice, charges; autodiff_backend=:ReverseDiff)
    println("ReverseDiff forces (compiled tape)")
    @btime ReverseDiff.gradient!($forces_reversediff, $compiled_energy_tape, $_positions)

    return nothing
end

benchmark_ewald(2)
    # Timings:
    # Energy without forces
    #   2.505 ms (75160 allocations: 3.69 MiB)
    # Manual forces
    #   9.217 ms (145079 allocations: 8.21 MiB)
    # FiniteDiff forces
    #   33.084 ms (901977 allocations: 44.26 MiB)
    # ForwardDiff forces
    #   3.780 ms (75174 allocations: 9.40 MiB)
    # ReverseDiff forces
    #   167.857 ms (3525296 allocations: 133.12 MiB)
    # ReverseDiff forces (compiled tape)
    #   65.120 ms (0 allocations: 0 bytes)

# benchmark_ewald(10) # ReverseDiff seems to crash Julia here (due to loop unrolling maybe? TODO)

#==========#
# Stresses # probably can be sped up further, but they work :)
#==========#

@time energy_ewald(lattice, charges, positions)
@time ForwardDiff.gradient(lattice -> energy_ewald(lattice, charges, positions), lattice)
@time FiniteDiff.finite_difference_gradient(lattice -> energy_ewald(lattice, charges, positions), lattice)

#==========#
# Misc     #
#==========#

# TODO 
# 1. Remove ForwardDiff from manual forces for better speed comparison
# 2. Try rewrite for more efficient reverse mode (ReverseDiff & Zygote)
#      - Use Distances.jl and its ChainRules ?
#      - Can we do something against the unrolling while-loop?
# x. (periodically check AbstractDifferentiation.jl)
