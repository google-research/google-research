using ProtoBuf

using BondLengthDistributions
using SmuUtilities
using dataset_pb2

export MatchingParameters
export SmuMolecule
export set_initial_score_and_incrementer!
export generate_search_state
export place_bonds!

"""Parameters controlling settable parameters during matching.
"""
mutable struct MatchingParameters
  # If true, all atoms must have their full valence satisfied.
  must_match_all_bonds::Bool
end
MatchingParameters() = MatchingParameters(true)

"""Description of a molecule used for building different bonding patterns.
"""
mutable struct SmuMolecule
  # The starting bond_topology, optionally with Hydrogens, and other singly
  # bonded atoms attached.
  starting_bond_topology::BondTopology

  # For each atom, the maximum number of bonds that can be attached.
  max_bonds::Vector{Int32}

  # During enumeration of possible molecules, we start with the
  # Hydrogen atom attached form of the BondTopology. That causes
  # atoms to have an initial number of bonds attached.
  bonds_with_hydrogens_attached::Vector{Int32}

  # During building a molecule, how many bonds are currently attached
  # to each atom
  current_bonds_attached::Vector{Int32}

  # Scores can be implemented as either multiplicative or additive forms.
  # Each needs an initial value and a function to accumulate the score.
  initial_score::Float32
  accumulate_score::Function

  # The `bonds` and `scores` are copied from the keys and values of
  # the bonds_to_scores dictionary given to the constructor.
  # The important thing is that these arrays are in the same order.
  bonds::Vector{Tuple{Int32, Int32}}
  scores::Vector{Vector{Float32}}

  # If true, then a successful match is only reported if each atom has the
  # maximum number of bonds attached.
  must_match_all_bonds::Bool

  function SmuMolecule(hydrogens_attached::BondTopology, 
              bonds_to_scores::Dict{Tuple{Int32,Int32}, Vector{Float32}},
              matching_parameters::MatchingParameters)
    mol = new();
    mol.starting_bond_topology = hydrogens_attached;
    # Delegate processing to a separate function, keep this small.
    smu_molecule_constructor!(mol);

    # separately extract the bonds and scores
    mol.bonds = collect(keys(bonds_to_scores));
    mol.scores = collect(values(bonds_to_scores));
    mol.must_match_all_bonds = matching_parameters.must_match_all_bonds
    mol
  end
end

"""Called from the SmuMolecule inner constructor as a regular function.
Initializes many of the attributes of `mol`.
"""
function smu_molecule_constructor!(mol::SmuMolecule)
  mol.accumulate_score = Base.:+
  mol.initial_score = 0.0
  natoms = length(mol.starting_bond_topology.atoms);
  mol.bonds_with_hydrogens_attached = zeros(Int32, natoms);
  mol.max_bonds = zeros(Int32, natoms);
  for i in 1:natoms
    mol.max_bonds[i] = SmuUtilities.smu_atom_type_to_max_con(mol.starting_bond_topology.atoms[i])
  end;
  # And during molecule building the current state.
  mol.current_bonds_attached = zeros(Int32, natoms);

  hasproperty(mol.starting_bond_topology, :bonds) || return
  for bond in mol.starting_bond_topology.bonds
    mol.bonds_with_hydrogens_attached[bond.atom_a + 1] += 1
    mol.bonds_with_hydrogens_attached[bond.atom_b + 1] += 1
  end
end

function set_initial_score_and_incrementer!(score::T, op, smu_molecule::SmuMolecule) where{T<:Number}
  smu_molecule.initial_score = score
  smu_molecule.accumulate_score = op
end

function accumulate_score(mol::SmuMolecule, existing_score::T, increment::T)::Float32 where {T<:Number}
  return mol.accumulate_score(existing_score, increment)
end

"""For each pair of atoms, return a list of plausible bond types.

The result will be used to enumerate all the possible bonding forms.
The resulting Vector has the same length as mol.bonds, and each item
in that vector is avector of the indices of the plausible bond types
for that connection.
Args:
  mol:
Returns:
  Vector of Vectors - one for each atom pair.
"""
function generate_search_state(mol::SmuMolecule)::Vector{Vector{Int32}}
  result = Vector{Vector{Int32}}()
  for ndx in 1:length(mol.bonds)
    # For each pair of atoms, the plausible bond types - non zero score.
    push!(result, findall(x->x > 0.0, mol.scores[ndx]))
  end

  return result
end

"""Possibly add a new bond to the current config.

If the bond can be placed, updates self._current_bonds_attached for
both `a`` and `a2`.
  Args:
    a1:
    a2:
    btype:
  Returns:
    true always - even if no bond is added. Perhaps change to return nothing.
print(f"Trying to place bond {btype} current {self._current_bonds_attached[a1]} and {self._current_bonds_attached[a2]}")
"""
function _place_bond!(a1::Int32, a2::Int32, btype, mol::SmuMolecule)::Bool
  btype > 0 || return true
  mol.current_bonds_attached[a1] + btype > mol.max_bonds[a1] && return false
  mol.current_bonds_attached[a2] + btype > mol.max_bonds[a2] && return false

  mol.current_bonds_attached[a1] += btype
  mol.current_bonds_attached[a2] += btype
  return true
end

"""Place bonds corresponding to `state`.

Args:
  state: for each pair of atoms, the kind of bond to be placed.
Returns:
  If successful, the score.
"""
function place_bonds!(state::Tuple,
                      mol::SmuMolecule)::Union{dataset_pb2.BondTopology, Nothing}
  result = dataset_pb2.BondTopology()  # To be returned.
  copy!(result, mol.starting_bond_topology)  # Only Hydrogens attached.
  result.score = mol.initial_score

  # Initialize state in mol to Hydrogens attached.
  mol.current_bonds_attached = copy(mol.bonds_with_hydrogens_attached)

  for i in 1:length(state)
    a1 = mol.bonds[i][1]
    a2 = mol.bonds[i][2]
    btype = state[i]
    _place_bond!(a1, a2, btype - 1, mol) || return nothing
    # If the bond is anything other than BOND_UNDEFINED, add it to result.
    (btype - 1) == BondTopology_BondType.BOND_UNDEFINED || add_bond!(a1 - 1, a2 - 1, btype - 1, result)
    result.score = accumulate_score(mol, result.score, mol.scores[i][btype])
  end

  # Optionally check whether all bonds have been matched
  mol.must_match_all_bonds || return result

  @debug("Cf bonds $(mol.current_bonds_attached) $(mol.max_bonds)")
  mol.current_bonds_attached == mol.max_bonds ? result : nothing
end
