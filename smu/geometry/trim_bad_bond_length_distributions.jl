"""When using distance geometry to generate geometries, lots of bad
distances get produced which messes up the resulting distributions.
Trim a series of bond length distribution files to a given frction of
the data.

This is a quick hack to trim those files. Not robust and will not
be needed when we have proper distributions"""

using ArgMacros
using CSV
using DataFrames
using LinearAlgebra

"""Very simplistic trimming of rhs and lhs based on the `counts` vector.
Args:
  distance: vector of distance values
  counts: vector of counts 
  fraction: what fraction of data to retain
  output_fname: where to write the result
"""
function process_file_contents(distance::Vector{Float64},
                      counts::Vector{Int64},
                      fraction::Real,
                      output_fname::String)::Bool
  ntot = sum(counts)
  if ntot < 100
    @warn("Too few points $(ntot), skipping")
    return true
  end
  nkeep = round(Int, fraction * ntot)
  @info("N = $(ntot) fraction $fraction NKeep $(nkeep)")
  lhs = 1
  rhs = length(counts)
  removed = 0
  to_remove = ntot - nkeep
  while removed  < to_remove
    # If equal counts, make an arbitrary decision.
    if counts[lhs] <= counts[rhs]
      removed += counts[lhs]
      lhs += 1
    else
      removed += counts[rhs]
      rhs -= 1
    end
  end
  while counts[rhs] == 0
    rhs -= 1
  end
  while counts[lhs] == 0
    lhs += 1
  end
  x = distance[lhs:rhs]
  y = counts[lhs:rhs]
  to_write = DataFrame(distance=x, count=y)
  CSV.write(output_fname, to_write, writeheader=false)
  return true
end

function process_file(input_fname::String, output_fname::String,
                      fraction::Float64)::Bool
  if ! isfile(input_fname)
    @warn("Skipping $input_fname")
    return true
  end
  @info("Processing $input_fname")
  df = CSV.File(input_fname) |> DataFrame
  return process_file_contents(df[!, 1], df[!, 2], fraction, output_fname)
end

function main()
  @inlinearguments begin
    @argumentrequired String input_stem "-i" "--input_stem"
    @argumentrequired String output_stem "-o" "--output_stem"
    @argumentrequired Float64 fraction "-f" "--fraction"
  end

  bstart = 0
  bond_types = [0, 1, 2, 3]

  atomic_numbers = [1, 6, 7, 8, 9]
  for (t1, t2) in filter(x->x[1] <= x[2], collect(Iterators.product(ntuple(i->atomic_numbers, 2)...))[:])
    for btype in bstart:3
      input_fname = "$input_stem.$(t1).$(btype).$(t2)"
      output_fname = "$output_stem.$(t1).$(btype).$(t2)"
      process_file(input_fname, output_fname, fraction)
    end
  end
end


main()
