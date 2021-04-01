using LinearAlgebra
using SpecialFunctions: erfc
using StaticArrays
using Test
using Zygote
using FiniteDiff

const Mat3{T} = SMatrix{3, 3, T, 9} where T
const Vec3{T} = SVector{3, T} where T
const MVec3{T} = MVector{3, T} where T

"""
A modified ewald summation suitable for AD with Zygote.
Original docstring below.

Compute the electrostatic interaction energy per unit cell between point
charges in a uniform background of compensating charge to yield net
neutrality. the `lattice` and `recip_lattice` should contain the
lattice and reciprocal lattice vectors as columns. `charges` and
`positions` are the point charges and their positions (as an array of
arrays) in fractional coordinates.
"""
function energy_ewald_zygote_faster(lattice, charges, positions; η=nothing)
    T = eltype(lattice)

    for i=1:3
        if norm(lattice[:,i]) == 0
            ## TODO should something more clever be done here? For now
            ## we assume that we are not interested in the Ewald
            ## energy of non-3D systems
            return T(0)
        end
    end
    energy_ewald_zygote_faster(lattice, T(2π) * inv(lattice'), charges, positions; η=η)
end

function energy_ewald_zygote_faster(lattice, recip_lattice, charges, positions; η=nothing)
    T = eltype(lattice)
    @assert T == eltype(recip_lattice)
    @assert length(charges) == length(positions)
    if η === nothing
        # Balance between reciprocal summation and real-space summation
        # with a slight bias towards reciprocal summation
        η = sqrt(sqrt(T(1.69) * norm(recip_lattice ./ 2T(π)) / norm(lattice))) / 2
    end

    #
    # Numerical cutoffs
    #
    # The largest argument to the exp(-x) function to obtain a numerically
    # meaningful contribution. The +5 is for safety.
    max_exponent = -log(eps(T)) + 5

    # The largest argument to the erfc function for various precisions.
    # To get an idea:
    #   erfc(5) ≈ 1e-12,  erfc(8) ≈ 1e-29,  erfc(10) ≈ 2e-45,  erfc(14) ≈ 3e-87
    max_erfc_arg = get(
        Dict(Float32 => 5, Float64 => 8, BigFloat => 14),
        T,
        something(findfirst(arg -> 100 * erfc(arg) < eps(T), 1:100), 100) # fallback for not yet implemented cutoffs
    )

    #
    # Reciprocal space sum
    #
    # Initialize reciprocal sum with correction term for charge neutrality
    sum_recip = - (sum(charges)^2 / 4η^2)

    # Function to return the indices corresponding
    # to a particular shell
    # TODO switch to an O(N) implementation
    function shell_indices(ish)
        (Vec3(i,j,k) for i in -ish:ish for j in -ish:ish for k in -ish:ish
        if maximum(abs.((i,j,k))) == ish)
    end

    # Loop over reciprocal-space shells
    gsh = 1 # Exclude G == 0
    any_term_contributes = true
    while any_term_contributes
        any_term_contributes = false

        # Compute G vectors and moduli squared for this shell patch
        for G in shell_indices(gsh)
            Gsq = sum(abs2, recip_lattice * Zygote.dropgrad(G))

            # Check if the Gaussian exponent is small enough
            # for this term to contribute to the reciprocal sum
            exponent = Gsq / 4η^2
            if exponent > max_exponent
                continue
            end

            cos_strucfac = sum(Z * cos(2T(π) * dot(r, Zygote.dropgrad(G))) for (r, Z) in zip(positions, charges))
            sin_strucfac = sum(Z * sin(2T(π) * dot(r, Zygote.dropgrad(G))) for (r, Z) in zip(positions, charges))
            sum_strucfac = cos_strucfac^2 + sin_strucfac^2

            any_term_contributes = true
            sum_recip += sum_strucfac * exp(-exponent) / Gsq

        end
        gsh += 1
    end
    # Amend sum_recip by proper scaling factors:
    sum_recip *= 4T(π) / abs(det(lattice))

    #
    # Real-space sum
    #
    # Initialize real-space sum with correction term for uniform background
    sum_real = Zygote.dropgrad(-2η / sqrt(T(π)) * sum(abs2, charges))

    # Loop over real-space shells
    rsh = 0 # Include R = 0
    any_term_contributes = true
    while any_term_contributes || rsh <= 1
        any_term_contributes = false

        # Loop over R vectors for this shell patch
        for R in shell_indices(rsh)
            for i = 1:length(positions), j = 1:length(positions)
                ti = positions[i]
                Zi = charges[i]
                tj = positions[j]
                Zj = charges[j]

                # Avoid self-interaction
                if rsh == 0 && ti == tj
                    continue
                end

                dist = norm(lattice * (ti - tj - Zygote.dropgrad(R)))

                # erfc decays very quickly, so cut off at some point
                if η * dist > max_erfc_arg
                    continue
                end

                any_term_contributes = true
                sum_real += Zi * Zj * erfc(η * dist) / dist
            end # i,j
        end # R
        rsh += 1
    end
    energy = (sum_recip + sum_real) / 2  # Divide by 2 (because of double counting)
    energy
end

#=======#

function compute_forces_zygote(positions, lattice, charges)
    _positions = reduce(hcat, positions)
    forces = first(
        Zygote.gradient(_positions -> begin
                positions = collect(eachcol(_positions))
                -energy_ewald_zygote_faster(lattice, charges, positions)
            end,
            _positions
        )
    )
    return forces
end

function compute_forces_finitediff(positions, lattice, charges)
    _positions = reduce(hcat, positions)
    forces = FiniteDiff.finite_difference_gradient(_positions -> begin
            positions = collect(eachcol(_positions))
            -energy_ewald_zygote_faster(lattice, charges, positions)
        end,
        _positions
    )
    return forces
end

#===#

function benchmark_ewald_zygote(num_atoms=20)
    lattice = Mat3(
        [0.0  5.131570667152971 5.131570667152971;
        5.131570667152971 0.0 5.131570667152971;
        5.131570667152971 5.131570667152971  0.0]
    )
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20 for _ in 1:num_atoms]
    charges = [14 for _ in 1:num_atoms]

    forces_finitediff = compute_forces_finitediff(positions, lattice, charges)
    forces_zygote = compute_forces_zygote(positions, lattice, charges)

    # check consistency of forces
    # @test forces_zygote ≈ forces_finitediff atol=1e-6
    @show sum(abs.(forces_finitediff - forces_zygote)) / (3num_atoms)

    println("Timings:")
    println("Energy without forces")
    @time energy_ewald_zygote(lattice, charges, positions)
    println("FiniteDiff forces")
    @time compute_forces_finitediff(positions, lattice, charges)
    println("Zygote forces")
    @time compute_forces_zygote(positions, lattice, charges)

    return nothing
end

# benchmark_ewald_zygote(2)
    # Timings:
    # Energy without forces
    #   0.004692 seconds (113.77 k allocations: 4.967 MiB)
    # FiniteDiff forces
    #   0.075556 seconds (1.37 M allocations: 59.604 MiB)
    # Zygote forces
    #   0.318643 seconds (1.03 M allocations: 48.197 MiB, 11.16% gc time)

# benchmark_ewald_zygote(10)
    # Timings:
    # Energy without forces
    #   0.285002 seconds (2.46 M allocations: 105.066 MiB, 57.46% gc time)
    # FiniteDiff forces
    #   8.808988 seconds (147.72 M allocations: 6.156 GiB, 18.05% gc time)
    # Zygote forces
    #   4.530250 seconds (15.03 M allocations: 717.221 MiB, 13.84% gc time)


    
# Zygote-rewrite required us to
# 1. do not use try-catch
# 2. use Zygote.@ignore on shell_indices
# 3. use Zygote.dropgrad on shell int vectors



const lattice = Mat3(
    [0.0  5.131570667152971 5.131570667152971;
    5.131570667152971 0.0 5.131570667152971;
    5.131570667152971 5.131570667152971  0.0]
)
# perturb positions away from equilibrium to get nonzero force
const positions = [Vec3(ones(3)/8+rand(3)/20) for _ in 1:2]
const charges = [14 for _ in 1:2]

@time energy_ewald_zygote_faster(lattice, charges, positions) # 0.319368 seconds (470.33 k allocations: 24.824 MiB, 6.28% gc time, 99.67% compilation time)

using BenchmarkTools
# @btime energy_ewald_zygote_faster($lattice, $charges, $positions) # 555.540 μs (3378 allocations: 53.33 KiB)

# @time energy_ewald(lattice, charges, positions)
# @time energy_ewald(lattice, charges, positions)

#========#
# A larger physical system with pymatgen
using DFTK

function benchmark_ewald_zygote_pymatgen(num_atoms=20)
    # lattice = Mat3(
    #     [0.0  5.131570667152971 5.131570667152971;
    #     5.131570667152971 0.0 5.131570667152971;
    #     5.131570667152971 5.131570667152971  0.0]
    # )
    # # perturb positions away from equilibrium to get nonzero force
    # positions = [ones(3)/8+rand(3)/20 for _ in 1:num_atoms]
    # charges = [14 for _ in 1:num_atoms]

    a = 10.263141334305942  # Lattice constant in Bohr
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]];
    pystruct = pymatgen_structure(lattice, atoms)
    pystruct.make_supercell([1, 1, 1])
    lattice = load_lattice(pystruct)
    positions = [s.frac_coords for s in pystruct.sites];
    charges = [14 for _ in 1:length(positions)]

    # forces_finitediff = compute_forces_finitediff(positions, lattice, charges)
    forces_zygote = compute_forces_zygote(positions, lattice, charges)

    # check consistency of forces
    # @test forces_zygote ≈ forces_finitediff atol=1e-6
    # @show sum(abs.(forces_finitediff - forces_zygote)) / (3num_atoms)

    println("Timings:")
    println("Energy without forces")
    @time energy_ewald_zygote(lattice, charges, positions)
    println("FiniteDiff forces")
    # @time compute_forces_finitediff(positions, lattice, charges)
    println("Zygote forces")
    @time compute_forces_zygote(positions, lattice, charges)

    return nothing
end

benchmark_ewald_zygote_pymatgen()
