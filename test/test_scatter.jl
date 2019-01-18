using Compat
using Test
using MPI

MPI.Init()

function scatter_array(A, root)
    comm = MPI.COMM_WORLD
    T = eltype(A)
    if MPI.Comm_rank(comm) == root
        global B = copy(A)
    else
        global B = Array{T}(undef,1)
    end
    C = MPI.Scatter(B, 1, root, comm)
end

comm = MPI.COMM_WORLD
root = 0

A = collect(1:MPI.Comm_size(comm))
B = scatter_array(A, root)
@test B[1] == MPI.Comm_rank(comm) + 1
for typ in Base.uniontypes(MPI.MPIDatatype)

    # Auto-allocated output
    global A = convert(Vector{typ},collect(1:MPI.Comm_size(comm)))
    global B = scatter_array(A, root)
    @test B[1] == convert(typ,MPI.Comm_rank(comm) + 1)

    # preallocated output
    if MPI.Comm_rank(MPI.COMM_WORLD) == root
        global B = copy(A)
    else
        global B = Array{typ}(undef, 1)
    end
    C = Array{typ}(undef, 1)
    MPI.Scatter!(B, C, 1, root, comm)
    @test C[1] == convert(typ,MPI.Comm_rank(comm) + 1)

    # MPI_IN_PLACE
end

MPI.Finalize()
