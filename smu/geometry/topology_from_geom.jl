using Combinatorics

using dataset_pb2

using BondLengthDistributions
using SmuMolecules
using SmuUtilities


const THRESHOLD = 2.0

"""Generate a BondTopology that joins each Hydrogen atom to its nearest
    heavy atom.
Args:
  bond_topology:
  distances:
Returns:
"""
function hydrogen_to_nearest_atom(bond_topology::dataset_pb2.BondTopology,
                             distances::Array{Float32,2})::Union{dataset_pb2.BondTopology, Nothing}
  result = dataset_pb2.BondTopology()
  # Not sure if this is safe or not.
  setproperty!(result, :atoms, bond_topology.atoms)

  single_bond = 1

  natoms = length(bond_topology.atoms)
  for a1 in 1:natoms
    bond_topology.atoms[a1] == dataset_pb2.BondTopology_AtomType.ATOM_H || continue

    shortest_distance = 1.0e+30
    closest_heavy_atom = -1
    for a2 in 1:natoms
      bond_topology.atoms[a2] == dataset_pb2.BondTopology_AtomType.ATOM_H && continue

      distances[a1, a2] >= THRESHOLD && continue

      if distances[a1, a2] < shortest_distance
        shortest_distance = distances[a1, a2]
        closest_heavy_atom = a2
      end
    end

    closest_heavy_atom < 0 && return nothing

    add_bond!(a1 - 1, closest_heavy_atom - 1, single_bond, result)
  end

  return result
end

# Enable sorting of BondTopology's
import Base: isless
Base.isless(bt1::BondTopology, bt2::BondTopology) = isless(bt1.score, bt2.score)

# Convenenience function for filtering scores.
nonzero = x->x>0.0

# Convenenience function for selecting non hydrogen atoms
not_hydrogen = x->x != BondTopology_AtomType.ATOM_H

"""Return all BondTopology's that are plausible.

  Given a molecule described by `bond_topology` and `geometry`, return all possible
  BondTopology that are consistent with that.
  Note that `bond_topology` will be put in a canonical form.
  Args:
    bond_length_distribution:
    bond_topology:
    geometry:
  Returns:
    TopologyMatches
"""
function bond_topologies_from_geom(
    bond_lengths::AllBondLengthDistributions,
    bond_topology::dataset_pb2.BondTopology,
    geometry::dataset_pb2.Geometry,
    matching_parameters::MatchingParameters)::dataset_pb2.TopologyMatches
  result = dataset_pb2.TopologyMatches()    # To be returned.
  length(bond_topology.atoms) == 1 && return result  # return an empty result

  canonical_bond_topology!(bond_topology)
  distances = SmuUtilities.distances(geometry)

  # First join each Hydrogen to its nearest heavy atom, thereby
  # creating a starting BondTopology from which all others can grow
  starting_bond_topology = hydrogen_to_nearest_atom(bond_topology, distances)
  starting_bond_topology === nothing && return result

  heavy_atom_indices = findall(not_hydrogen, bond_topology.atoms)
  length(heavy_atom_indices) < 2 && return result

  # For each atom pair, a list of possible bond types.
  # Key is a tuple of the two atom numbers, value is a Vector
  # with the score for each bond type.

  bonds_to_scores = Dict{Tuple{Int32,Int32}, Vector{Float32}}()

  for (i, j) in Combinatorics.combinations(heavy_atom_indices, 2)  # All pairs
    (dist = distances[i, j]) > THRESHOLD && continue

    scores = [pdf(bond_lengths, bond_topology.atoms[i], bond_topology.atoms[j], btype, dist)
              for btype in 0:4]

    any(nonzero, scores) && (bonds_to_scores[(i, j)] = scores)
  end

  @debug("bonds_to_scores $(bonds_to_scores)")
  isempty(bonds_to_scores) && return result

  found_topologies = Vector{BondTopology}()  # Will be set into `result`.

  mol = SmuMolecule(starting_bond_topology, bonds_to_scores, matching_parameters)

  search_space = generate_search_state(mol)
  for s in Iterators.product(search_space...)
    @debug("Placing state $s")
    (bt = place_bonds!(s, mol)) == nothing && continue
    is_single_fragment(bt) || continue

    canonical_bond_topology!(bt)
    same_bond_topology(bond_topology, bt) && (bt.is_starting_topology = true)
    # smiles not set
    push!(found_topologies, bt)
  end

  isempty(found_topologies) && return result

  length(found_topologies) > 1 && sort!(found_topologies, rev=true)

  setproperty!(result, :bond_topology, found_topologies)

  return result
end
