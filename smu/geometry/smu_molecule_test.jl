"""Testing the SmuMolecule.
The 1 based array indexing in Julia makes this somewhat discordant with the phython
implementation.
"""

module TestSmuMolecule

using ProtoBuf
using Test

using SmuMolecules
using SmuUtilities

using dataset_pb2

export all_tests

"""Make a BondTopology consisting of `n` disconnected Carbon atoms.
"""
function carbon_atoms(n::T)::BondTopology where {T<:Integer}
  result = BondTopology()
  for i in 1:n
    add_atom_by_atomic_number!(result, 6)
  end
  return result
end

"""The simplest multi-atom molecule, CC"""
function test_ethane()
  matching_parameters = MatchingParameters()
  matching_parameters.must_match_all_bonds = false

  ethane = carbon_atoms(2)
  add_bond!(0, 1, 1, ethane)
  scores = [0.1, 1.1, 2.1, 3.1]
  bonds_to_scores = Dict{Tuple{Int32, Int32}, Vector{Float32}}((1, 2) => scores)
  mol = SmuMolecule(ethane, bonds_to_scores, matching_parameters)
  state = generate_search_state(mol)
  @test length(state) == 1
  @test state == [[1, 2, 3, 4]]

  for s in Iterators.product(state...)
    res = place_bonds!(s, mol)
    @test res !== nothing
    # self.assertAlmostEqual(res.score, scores[i])
  end
  true
end

function test_ethane_all_btypes()
  matching_parameters = MatchingParameters()
  matching_parameters.must_match_all_bonds = false

  test_cases = [
    [0, dataset_pb2.BondTopology_BondType.BOND_UNDEFINED],
    [1, dataset_pb2.BondTopology_BondType.BOND_SINGLE],
    [2, dataset_pb2.BondTopology_BondType.BOND_DOUBLE],
    [3, dataset_pb2.BondTopology_BondType.BOND_TRIPLE],
  ]

  for t in test_cases
    numeric_btype = t[1]
    smu_btype = t[2]
    cc = carbon_atoms(2)
    bonds_to_scores = Dict{Tuple{Int32,Int32}, Vector{Float32}}((1, 2) => zeros(Float32, 4))
    bonds_to_scores[(1, 2)][numeric_btype + 1] = 1.0
    mol = SmuMolecule(cc, bonds_to_scores, matching_parameters)
    state = generate_search_state(mol)
    for s in Iterators.product(state...)
      res = place_bonds!(s, mol)
      @test res !== nothing
      if numeric_btype == 0
        @test !hasproperty(res, :bonds)
      else
        @test length(res.bonds) == 1
        @test res.bonds[1].bond_type == smu_btype
      end
    end
  end
  true
end

"""Checking to see if a BondTopology has the right number of bonds is
complicated by the fact that if there are no bonds present, the :bonds
attribute will be missing.
Args:
  bond_topology:
  expected_bonds: the number of bonds expected to be in `bond_topology`.
Returns:
  true if the number of bonds present matches `expected_bonds`.
"""
function bond_count_matches(bond_topology::BondTopology, expected_bonds)::Bool
  if !hasproperty(bond_topology, :bonds)
    return expected_bonds == 0
  end

  return length(bond_topology.bonds) == expected_bonds
end

struct A1A2BtypeScore
  btype1::Int32
  btype2::Int32
  expected_bonds::Int32
  expected_score::Union{Float32, Nothing}
end

function test_propane_all()
  matching_parameters = MatchingParameters()
  matching_parameters.must_match_all_bonds = false

  test_cases = [
    A1A2BtypeScore(0, 0, 0, 2.0),
    A1A2BtypeScore(0, 1, 1, 2.0),
    A1A2BtypeScore(0, 2, 1, 2.0),
    A1A2BtypeScore(0, 3, 1, 2.0),
    A1A2BtypeScore(1, 1, 2, 2.0),
    A1A2BtypeScore(1, 2, 2, 2.0),
    A1A2BtypeScore(1, 3, 2, 2.0),
    A1A2BtypeScore(2, 2, 2, 2.0),
    A1A2BtypeScore(2, 3, 0, nothing),
    A1A2BtypeScore(3, 3, 0, nothing)
  ]
  for t in test_cases
    btype1 = t.btype1
    btype2 = t.btype2
    expected_bonds = t.expected_bonds
    expected_score = t.expected_score
    cc = carbon_atoms(3)
    bonds_to_scores = Dict{Tuple{Int32,Int32}, Vector{Float32}}((1, 2) => zeros(Float32, 4),
                       (2, 3) => zeros(Float32, 4))
    bonds_to_scores[(1, 2)][btype1 + 1] = 1.0
    bonds_to_scores[(2, 3)][btype2 + 1] = 1.0
    mol = SmuMolecule(cc, bonds_to_scores, matching_parameters)
    state = generate_search_state(mol)
    for s in Iterators.product(state...)
      res = place_bonds!(s, mol)
      if expected_score !== nothing
        @test res !== nothing
        @test bond_count_matches(res, expected_bonds)
        @test isapprox(res.score, expected_score)
        if btype1 == 0
          if btype2 > 0
            @test res.bonds[1].bond_type == btype2
          end
        else
          @test Set((btype1, btype2)) == Set((res.bonds[1].bond_type, res.bonds[2].bond_type))
        end
      else
        @test res === nothing
      end
    end
  end
  true
end

function test_operators()
  matching_parameters = MatchingParameters()
  matching_parameters.must_match_all_bonds = false

  cc = carbon_atoms(3)
  add_bond!(0, 1, 1, cc)
  add_bond!(1, 2, 1, cc)
  bonds_to_scores = Dict{Tuple{Int32,Int32}, Vector{Float32}}((1, 2) => zeros(Float32, 4),
                                            (2, 3) => zeros(Float32, 4))
  scores = [1.0, 3.0]
  bonds_to_scores[(1, 2)][1] = scores[1]
  bonds_to_scores[(2, 3)][1] = scores[2]
  mol = SmuMolecule(cc, bonds_to_scores, matching_parameters)
  set_initial_score_and_incrementer!(1.0, Base.:*, mol)
  state = generate_search_state(mol)
  for s in Iterators.product(state...)
    res = place_bonds!(s, mol)
    @test isapprox(res.score, prod(scores))
  end
  true
end


function all_tests()
  @test test_ethane()
  @test test_ethane_all_btypes()
  @test test_propane_all()
  @test test_operators()
end



end  # module TestSmuMolecule
