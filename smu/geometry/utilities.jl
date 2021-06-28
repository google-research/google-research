# utility functions needed for discerning topology from geometry

using Combinatorics

using dataset_pb2

export bohr_to_angstroms, angstroms_to_bohr, distance_between_atoms, bonded, distances
export add_atom!, add_bond!
export smu_atype_to_atomic_number, number_to_smu_btype, smu_btype_to_number, smu_atom_type_to_max_con
export atomic_number_to_smu_atom_type
export add_atom_by_atomic_number!, add_atom_by_type!
export distances, bonded
export canonical_bond_topology!, same_bond_topology
export is_single_fragment

function bohr_to_angstroms(distance)
  return distance * 0.529177249
end

function angstroms_to_bohr(distance)
  return distance / 0.529177249
end

function distance_between_atoms(geom::dataset_pb2.Geometry, a1, a2)::Float32
  return bohr_to_angstroms(sqrt(
    (geom.atom_positions[a1].x - geom.atom_positions[a2].x) *
    (geom.atom_positions[a1].x - geom.atom_positions[a2].x) +
    (geom.atom_positions[a1].y - geom.atom_positions[a2].y) *
    (geom.atom_positions[a1].y - geom.atom_positions[a2].y) +
    (geom.atom_positions[a1].z - geom.atom_positions[a2].z) *
    (geom.atom_positions[a1].z - geom.atom_positions[a2].z)
  ))
end

_smu_atype_to_atomic_number = Dict{Int32, Int32}(
  BondTopology_AtomType.ATOM_H => 1,
  BondTopology_AtomType.ATOM_C => 6,
  BondTopology_AtomType.ATOM_N => 7,
  BondTopology_AtomType.ATOM_NPOS => 7,
  BondTopology_AtomType.ATOM_O => 8,
  BondTopology_AtomType.ATOM_ONEG => 8,
  BondTopology_AtomType.ATOM_F => 9
)

function smu_atype_to_atomic_number(smu_type::T)::Int32 where {T<:Integer}
  return _smu_atype_to_atomic_number[smu_type]
end

"""Given an atomic number and charge, return the BondTopology_AtomType
Args:
  atomic_number:
  charge:
Returns:
  BondTopology_AtomType
"""
function atomic_number_to_smu_atom_type(atomic_number::Integer, charge::Integer=0)::Int32
  if atomic_number == 6
    return BondTopology_AtomType.ATOM_C
  elseif atomic_number == 7
    if charge == 0
      return BondTopology_AtomType.ATOM_N
    else
      return BondTopology_AtomType.ATOM_NPOS
    end
  elseif atomic_number == 8
    if charge == 0
      return BondTopology_AtomType.ATOM_O
    else
      return BondTopology_AtomType.ATOM_ONEG
    end
  elseif atomic_number == 9
    return BondTopology_AtomType.ATOM_F
  elseif atomic_number == 1
    return BondTopology_AtomType.ATOM_H
  else
    @error("Unrecognized atomic number $(atomic_number)")
  end
end

_number_to_smu_btype = Dict{Int32, Int32}(
  0 => BondTopology_BondType.BOND_UNDEFINED,
  1 => BondTopology_BondType.BOND_SINGLE,
  2 => BondTopology_BondType.BOND_DOUBLE,
  3 => BondTopology_BondType.BOND_TRIPLE
)

function number_to_smu_btype(n::T)::Int32 where {T<:Integer}
  return _number_to_smu_btype[n]
end

_smu_btype_to_number = Dict{Int32, Int32}(
  BondTopology_BondType.BOND_UNDEFINED => 0,
  BondTopology_BondType.BOND_SINGLE => 1,
  BondTopology_BondType.BOND_DOUBLE => 2,
  BondTopology_BondType.BOND_TRIPLE => 3
)

_smu_atom_type_to_max_con = Dict{Int32, Int32}(
  BondTopology_AtomType.ATOM_H => 1,
  BondTopology_AtomType.ATOM_C => 4,
  BondTopology_AtomType.ATOM_N => 3,
  BondTopology_AtomType.ATOM_NPOS => 3,
  BondTopology_AtomType.ATOM_O => 2,
  BondTopology_AtomType.ATOM_ONEG => 1,
  BondTopology_AtomType.ATOM_F => 1
)

# Return the maximum number of bonds for `smu_atype`.
# Unfortunately the type declaration on the next line does not work.
# function smu_atom_type_to_max_con(smu_atype::BondTopology_AtomType)::Int32

function smu_atom_type_to_max_con(smu_atype::T)::Int32 where {T <:Integer}
  return _smu_atom_type_to_max_con[smu_atype]
end

function smu_btype_to_number(smu_btype::T)::Int32 where {T<:Integer}
  return _smu_btype_to_number[smu_btype]
end


function add_bond!(a1::T, a2::T, btype, bond_topology::dataset_pb2.BondTopology) where {T<:Integer}
  smu_btype = number_to_smu_btype(btype)

  new_bond = BondTopology_Bond(atom_a=a1, atom_b=a2, bond_type=smu_btype)

  if !hasproperty(bond_topology, :bonds)
    setproperty!(bond_topology, :bonds, [new_bond])
  else
    push!(bond_topology.bonds, new_bond)
  end
end

function add_atom_by_atomic_number!(bond_topology::dataset_pb2.BondTopology, atomic_number::T, charge::T=0) where {T<:Integer}
  smu_atype = atomic_number_to_smu_atom_type(atomic_number, charge)
  add_atom_by_type!(bond_topology, smu_atype)
end

function add_atom_by_type!(bond_topology::dataset_pb2.BondTopology, smu_atype::T) where {T <:Integer}
  if !hasproperty(bond_topology, :atoms)
    setproperty!(bond_topology, :atoms, [smu_atype])
  else
    push!(bond_topology.atoms, smu_atype)
  end
end

"""Return an int array of the bonded atoms in `bond_topology`.
The returned array is square[natoms,natoms]. If there is no bond
between the atoms, the value is zero. If the atoms are bonded,
the entry will be the bond type.
Args:
  bond_topology:
Returns:
  Vector of BondType's
"""
function bonded(bond_topology::dataset_pb2.BondTopology)::Array{Int32, 2}
  natoms = length(bond_topology.atoms)
  connected = zeros(Int32, (natoms, natoms))  # to be returned
  # Note need to convert to 1 indexing for Julia
  for bond in bond_topology.bonds
    a1 = bond.atom_a + 1
    a2 = bond.atom_b + 1
    connected[a1, a2] = connected[a2, a1] = bond.bond_type
  end
  return connected
end

"""Given a connection matrix `bonded`, return a vector of
vectors, one for each atom, containing the indices of the
bonded atoms.
Args:
  bonded:
Returns:
"""
function connections(bonded::Array{Int32, 2})::Vector{Vector{Int32}}
  natoms = size(bonded, 1)
  result = Vector{Vector{Int32}}(undef, (natoms))
  for i in 1:natoms
    nbrs = Vector{Int32}()
    for j in 1:natoms
      if bonded[i,j] > 0
        push!(nbrs, j)
      end
    end
    result[i] = nbrs
  end
  return result
end

"""Return a float array of the interatomic distances in `geometry`.
Args:
  geometry:
Returns:
  a numpy array of distances
"""
function distances(geometry)::Array{Float32, 2}
  natoms = length(geometry.atom_positions)
  result = zeros(Float32, (natoms, natoms))
  for atoms in combinations(1:natoms, 2)
    i = atoms[1]
    j = atoms[2]
    result[i, j] = result[j, i] = distance_between_atoms(geometry, i, j)
  end
  return result
end


"""Transform the bonds attribute of `bond_topology` to a canonical form.

Args:
  bond_topology:
Returns:
  BondTopology
"""
function canonical_bond_topology!(bond_topology)
  if length(bond_topology.bonds) < 2
    return
  end

  for bond in bond_topology.bonds
    if bond.atom_a > bond.atom_b
      bond.atom_a, bond.atom_b = bond.atom_b, bond.atom_a
    end
  end

  sort!(bond_topology.bonds, by = b -> (b.atom_a, b.atom_b))
end

Base.:(==)(b1::BondTopology_Bond, b2::BondTopology_Bond)::Bool = b1.atom_a == b2.atom_a &&
                b1.atom_b == b2.atom_b && b1.bond_type == b2.bond_type

"""Return True if bt1 == bt2.
Note that there is no attempt to canonialise the protos.
Args:
Returns:
"""
function same_bond_topology(bt1, bt2)::Bool
  length(bt2.atoms) == length(bt2.atoms) || return false
  length(bt2.bonds) == length(bt2.bonds) || return false

  # https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
  # all(y->y==x[1], x);

  atoms1 = bt1.atoms
  if ! all(atoms1->atoms1==bt2.atoms[1], bt2.atoms)
    return false
  end

  # This should work, but raises an exception.
  # bonds1 = bt1.bonds
  # bonds2 = bt2.bonds
  # return all(bonds1->bonds1==bonds2[1], bonds2)

  for i in 1:length(bt1.bonds)
    bt1.bonds[i] != bt2.bonds[i] && return false
  end
  return true
end

"""Recusrively visit nodes in the graph defined by `nbrs`.
Args:
  nbrs:
  atom:
  visited:
Returns:
  The number of nodes visited - including `atom`.
"""
function visit(nbrs::Vector, atom::T, visited::Vector{Bool})::Int32 where {T<:Integer}
  visited[atom] = true
  result = 1    # To be returned.
  for nbr in nbrs[atom]
    if visited[nbr] > 0
      continue
    end
    result += visit(nbrs, nbr, visited)
  end

  return result
end

"""Return True if `bond_topology` is a single fragment.
Args:
  bond_topology:
Returns:
  True if `bond_topology` is a single fragment.
"""
function is_single_fragment(bond_topology::BondTopology)::Bool
  natoms = length(bond_topology.atoms)
  nbonds = length(bond_topology.bonds)
  # Some special cases are easy.
  natoms == 1 && return true
  if natoms == 2 && nbonds == 1
    return true
  end
  if natoms == 3 && nbonds == 2
    return true
  end
  if natoms == nbonds && natoms <= 4
    return true
  end

  attached = connections(bonded(bond_topology))
  # Any atom with zero neighbors means a detached atom.
  any(n->length(n) == 0, attached) && return false

  visited = zeros(Bool, natoms)
  # Mark anything with a single connection as visited.
  # Record the index of an atom that has multiple connections.
  a_multiply_connected_atom = -1
  for i in 1:natoms
    if bond_topology.atoms[i] == BondTopology_AtomType.ATOM_H
      visited[i] = 1
      continue
    end

    if length(attached[i]) > 1
      a_multiply_connected_atom = i
      continue
    end

    # A singly connected heavy atom. Mark visited if not part of a two atom fragment.
    if length(attached[attached[i][1]]) > 1
      visited[i] = 1
    end
  end

  # Not sure this can happen.
  a_multiply_connected_atom < 0 && return false

  number_visited = sum(visited) + visit(attached, a_multiply_connected_atom, visited)
  return number_visited == natoms
end
