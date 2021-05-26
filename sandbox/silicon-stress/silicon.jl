# Very basic setup, useful for testing
using DFTK
using LinearAlgebra

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [4, 4, 4]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-8)

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
E = kinetic_energy(lattice)

# diff E wrt lattice

# works fine
using ForwardDiff
@time ForwardDiff.gradient(kinetic_energy, lattice) # 3.627630 seconds (5.99 M allocations: 363.981 MiB, 5.08% gc time, 98.69% compilation time)

# works but long compile time and gives ComplexF64 results
# hypothesis: slow compilation due to loop (and generator) unrolling
using Zygote
@time Zygote.gradient(kinetic_energy, lattice) # 61.094425 seconds (63.31 M allocations: 3.715 GiB, 3.85% gc time, 67.43% compilation time)

# works fine
using ReverseDiff
@time ReverseDiff.gradient(kinetic_energy, lattice) # 5.409118 seconds (9.60 M allocations: 516.091 MiB, 14.61% gc time, 89.56% compilation time)

using BenchmarkTools
@btime ForwardDiff.gradient(kinetic_energy, lattice) #   3.152 ms (   7681 allocations:  11.00 MiB)
@btime Zygote.gradient(kinetic_energy, lattice)      #  99.395 ms ( 358600 allocations: 129.91 MiB)
@btime ReverseDiff.gradient(kinetic_energy, lattice) # 278.207 ms (4301696 allocations: 158.10 MiB)

using FiniteDiff
@time FiniteDiff.finite_difference_gradient(kinetic_energy, lattice)
@btime FiniteDiff.finite_difference_gradient(kinetic_energy, lattice) #   10.901 ms (136658 allocations: 30.14 MiB)

#===#
# Can we compare to differentiating *through* the SCF solve ?
function kinetic_energy(lattice, basis)
    scfres = self_consistent_field(basis, tol=1e-8)
    kinetic_energy(lattice, basis, scfres.ψ, scfres.occupation)
end

@time kinetic_energy(lattice, basis) # 6.563267 seconds (237.02 k allocations: 422.728 MiB, 1.07% gc time)
@time ForwardDiff.gradient(lattice -> kinetic_energy(lattice, basis), lattice) # 10.585561 seconds (5.96 M allocations: 801.731 MiB, 2.79% gc time, 41.07% compilation time)
