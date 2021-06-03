# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using BenchmarkTools

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15          # kinetic energy cutoff in Hartree -- can increase to make G_vectors larger (larger solve time)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

@time scfres = self_consistent_field(basis, tol=1e-8) # 75.068789 seconds (138.55 M allocations: 8.145 GiB, 4.59% gc time, 24.68% compilation time)

# TODO try to rewrite for Zygote
# e.g. translate loops to dense arrays or maps (?)

function kinetic_energy(lattice, basis, ψ, occ)
    recip_lattice = 2π * inv(lattice')
    E = zero(Float64)
    kinetic_energies = [[sum(abs2, recip_lattice * (G + kpt.coordinate)) / 2
                         for G in  G_vectors(kpt)]
                        for kpt in basis.kpoints]
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            E += (basis.kweights[ik] * occ[ik][iband]
                  * real(dot(ψnk, kinetic_energies[ik] .* ψnk)))
        end
    end
    E
end
kinetic_energy(lattice) = kinetic_energy(lattice, basis, scfres.ψ, scfres.occupation)

@time E = kinetic_energy(lattice)
@btime kinetic_energy(lattice) # 648.802 μs (7590 allocations: 1.67 MiB)

# stress := diff E wrt lattice

#===#
# Check results and compile times on first call
stresses = Dict()

# works fine
using ForwardDiff
@time stresses[:ForwardDiff] = ForwardDiff.gradient(kinetic_energy, lattice) # 3.627630 seconds (5.99 M allocations: 363.981 MiB, 5.08% gc time, 98.69% compilation time)

# works but long compile time and gives ComplexF64 results
# hypothesis: slow compilation due to loop (and generator) unrolling
using Zygote
@time stresses[:Zygote] = Zygote.gradient(kinetic_energy, lattice) # 61.094425 seconds (63.31 M allocations: 3.715 GiB, 3.85% gc time, 67.43% compilation time)

# works fine
using ReverseDiff
@time stresses[:ReverseDiff] = ReverseDiff.gradient(kinetic_energy, lattice) # 5.409118 seconds (9.60 M allocations: 516.091 MiB, 14.61% gc time, 89.56% compilation time)

# sanity check
using FiniteDiff
@time stresses[:FiniteDiff] = FiniteDiff.finite_difference_gradient(kinetic_energy, lattice) # 2.606210 seconds (2.87 M allocations: 232.911 MiB, 19.92% gc time, 99.19% compilation time)

stresses
# Dict{Any, Any} with 4 entries:
#   :ForwardDiff => [0.206717 -0.207215 -0.207228; -0.198914 0.209309 -0.20982; -0.197421 -0.208313 0.207815]
#   :FiniteDiff  => [0.206717 -0.207215 -0.207228; -0.198914 0.209309 -0.20982; -0.197421 -0.208313 0.207815]
#   :Zygote      => (ComplexF64[0.206717+2.92541e-19im -0.207215-1.57175e-19im -0.207228-7.86273e-19im; -0.198914+1.08911e-19im 0.209309+4.47104e-20im -0.20982-5.38442e-19im; -0.197421-6.72652e-19im -0.2…
#   :ReverseDiff => [0.206717 -0.207215 -0.207228; -0.198914 0.209309 -0.20982; -0.197421 -0.208313 0.207815]

@btime ForwardDiff.gradient(kinetic_energy, lattice) #   3.152 ms (   7681 allocations:  11.00 MiB)
@btime Zygote.gradient(kinetic_energy, lattice)      #  99.395 ms ( 358600 allocations: 129.91 MiB)
@btime ReverseDiff.gradient(kinetic_energy, lattice) # 278.207 ms (4301696 allocations: 158.10 MiB)
@btime FiniteDiff.finite_difference_gradient(kinetic_energy, lattice) # 10.901 ms (136658 allocations: 30.14 MiB)
