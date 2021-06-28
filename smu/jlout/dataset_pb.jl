# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ProtoBuf.google.protobuf

const AvailabilityEnum = (;[
    Symbol("UNSPECIFIED") => Int32(0),
    Symbol("INTERNAL_ONLY") => Int32(1),
    Symbol("STANDARD") => Int32(2),
    Symbol("COMPLETE") => Int32(3),
]...)

const BondTopology_AtomType = (;[
    Symbol("ATOM_UNDEFINED") => Int32(0),
    Symbol("ATOM_C") => Int32(1),
    Symbol("ATOM_N") => Int32(2),
    Symbol("ATOM_NPOS") => Int32(3),
    Symbol("ATOM_O") => Int32(4),
    Symbol("ATOM_ONEG") => Int32(5),
    Symbol("ATOM_F") => Int32(6),
    Symbol("ATOM_H") => Int32(7),
]...)

const BondTopology_BondType = (;[
    Symbol("BOND_UNDEFINED") => Int32(0),
    Symbol("BOND_SINGLE") => Int32(1),
    Symbol("BOND_DOUBLE") => Int32(2),
    Symbol("BOND_TRIPLE") => Int32(3),
]...)

mutable struct BondTopology_Bond <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BondTopology_Bond(; kwargs...)
        obj = new(meta(BondTopology_Bond), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct BondTopology_Bond
const __meta_BondTopology_Bond = Ref{ProtoMeta}()
function meta(::Type{BondTopology_Bond})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BondTopology_Bond)
            __meta_BondTopology_Bond[] = target = ProtoMeta(BondTopology_Bond)
            allflds = Pair{Symbol,Union{Type,String}}[:atom_a => Int32, :atom_b => Int32, :bond_type => Int32]
            meta(target, BondTopology_Bond, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BondTopology_Bond[]
    end
end
function Base.getproperty(obj::BondTopology_Bond, name::Symbol)
    if name === :atom_a
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :atom_b
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :bond_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct BondTopology <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BondTopology(; kwargs...)
        obj = new(meta(BondTopology), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct BondTopology
const __meta_BondTopology = Ref{ProtoMeta}()
function meta(::Type{BondTopology})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BondTopology)
            __meta_BondTopology[] = target = ProtoMeta(BondTopology)
            pack = Symbol[:atoms]
            allflds = Pair{Symbol,Union{Type,String}}[:atoms => Base.Vector{Int32}, :bonds => Base.Vector{BondTopology_Bond}, :smiles => AbstractString, :bond_topology_id => Int32, :score => Float32, :is_starting_topology => Bool]
            meta(target, BondTopology, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BondTopology[]
    end
end
function Base.getproperty(obj::BondTopology, name::Symbol)
    if name === :atoms
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :bonds
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{BondTopology_Bond}
    elseif name === :smiles
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :bond_topology_id
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :score
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :is_starting_topology
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct TopologyMatches <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function TopologyMatches(; kwargs...)
        obj = new(meta(TopologyMatches), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct TopologyMatches
const __meta_TopologyMatches = Ref{ProtoMeta}()
function meta(::Type{TopologyMatches})
    ProtoBuf.metalock() do
        if !isassigned(__meta_TopologyMatches)
            __meta_TopologyMatches[] = target = ProtoMeta(TopologyMatches)
            allflds = Pair{Symbol,Union{Type,String}}[:bond_topology => Base.Vector{BondTopology}]
            meta(target, TopologyMatches, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_TopologyMatches[]
    end
end
function Base.getproperty(obj::TopologyMatches, name::Symbol)
    if name === :bond_topology
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{BondTopology}
    else
        getfield(obj, name)
    end
end

mutable struct Geometry_AtomPos <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Geometry_AtomPos(; kwargs...)
        obj = new(meta(Geometry_AtomPos), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Geometry_AtomPos
const __meta_Geometry_AtomPos = Ref{ProtoMeta}()
function meta(::Type{Geometry_AtomPos})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Geometry_AtomPos)
            __meta_Geometry_AtomPos[] = target = ProtoMeta(Geometry_AtomPos)
            allflds = Pair{Symbol,Union{Type,String}}[:x => Float32, :y => Float32, :z => Float32]
            meta(target, Geometry_AtomPos, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Geometry_AtomPos[]
    end
end
function Base.getproperty(obj::Geometry_AtomPos, name::Symbol)
    if name === :x
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :y
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :z
        return (obj.__protobuf_jl_internal_values[name])::Float32
    else
        getfield(obj, name)
    end
end

mutable struct Geometry <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Geometry(; kwargs...)
        obj = new(meta(Geometry), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Geometry
const __meta_Geometry = Ref{ProtoMeta}()
function meta(::Type{Geometry})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Geometry)
            __meta_Geometry[] = target = ProtoMeta(Geometry)
            allflds = Pair{Symbol,Union{Type,String}}[:atom_positions => Base.Vector{Geometry_AtomPos}]
            meta(target, Geometry, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Geometry[]
    end
end
function Base.getproperty(obj::Geometry, name::Symbol)
    if name === :atom_positions
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Geometry_AtomPos}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_Errors <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_Errors(; kwargs...)
        obj = new(meta(Properties_Errors), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_Errors
const __meta_Properties_Errors = Ref{ProtoMeta}()
function meta(::Type{Properties_Errors})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_Errors)
            __meta_Properties_Errors[] = target = ProtoMeta(Properties_Errors)
            fnum = Int[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,65]
            allflds = Pair{Symbol,Union{Type,String}}[:error_frequencies => Int32, :error_atomic_analysis => Int32, :error_nmr_analysis_b3lyp_small_basis => Int32, :error_nmr_analysis_b3lyp_large_basis => Int32, :error_nmr_analysis_pbe0_small_basis => Int32, :error_nmr_analysis_pbe0_large_basis => Int32, :error_charge_analysis => Int32, :error_energies_orbitals_pvtz => Int32, :error_energies_orbitals_pvqz => Int32, :error_energies_orbitals_pcvtz => Int32, :error_excitation_energies => Int32, :error_single_point_energies => Int32, :error_inconsistent_molecule_energy_turbomole_mrcc => Int32, :error_inconsistent_cation_energy_turbomole_mrcc => Int32, :error_inconsistent_molecule_energy_turbomole_orca => Int32, :error_inconsistent_cation_energy_turbomole_orca => Int32, :error_normal_modes => Int32, :error_rotational_modes => Int32, :error_nsvho1 => Int32, :error_nsvho2 => Int32, :error_nsvho3 => Int32, :error_nsvneg => Int32, :error_nstat1 => Int32, :error_nstatc => Int32, :error_nstatt => Int32, :error_nsvego => Int32, :error_nsvg09 => Int32, :error_during_merging => AbstractString]
            meta(target, Properties_Errors, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_Errors[]
    end
end
function Base.getproperty(obj::Properties_Errors, name::Symbol)
    if name === :error_frequencies
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_atomic_analysis
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nmr_analysis_b3lyp_small_basis
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nmr_analysis_b3lyp_large_basis
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nmr_analysis_pbe0_small_basis
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nmr_analysis_pbe0_large_basis
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_charge_analysis
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_energies_orbitals_pvtz
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_energies_orbitals_pvqz
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_energies_orbitals_pcvtz
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_excitation_energies
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_single_point_energies
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_inconsistent_molecule_energy_turbomole_mrcc
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_inconsistent_cation_energy_turbomole_mrcc
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_inconsistent_molecule_energy_turbomole_orca
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_inconsistent_cation_energy_turbomole_orca
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_normal_modes
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_rotational_modes
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nsvho1
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nsvho2
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nsvho3
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nsvneg
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nstat1
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nstatc
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nstatt
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nsvego
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_nsvg09
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :error_during_merging
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct Properties_StringMolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_StringMolecularProperty(; kwargs...)
        obj = new(meta(Properties_StringMolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_StringMolecularProperty
const __meta_Properties_StringMolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_StringMolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_StringMolecularProperty)
            __meta_Properties_StringMolecularProperty[] = target = ProtoMeta(Properties_StringMolecularProperty)
            allflds = Pair{Symbol,Union{Type,String}}[:value => AbstractString]
            meta(target, Properties_StringMolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_StringMolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_StringMolecularProperty, name::Symbol)
    if name === :value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct Properties_ScalarMolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_ScalarMolecularProperty(; kwargs...)
        obj = new(meta(Properties_ScalarMolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_ScalarMolecularProperty
const __meta_Properties_ScalarMolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_ScalarMolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_ScalarMolecularProperty)
            __meta_Properties_ScalarMolecularProperty[] = target = ProtoMeta(Properties_ScalarMolecularProperty)
            allflds = Pair{Symbol,Union{Type,String}}[:value => Float64]
            meta(target, Properties_ScalarMolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_ScalarMolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_ScalarMolecularProperty, name::Symbol)
    if name === :value
        return (obj.__protobuf_jl_internal_values[name])::Float64
    else
        getfield(obj, name)
    end
end

mutable struct Properties_MultiScalarMolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_MultiScalarMolecularProperty(; kwargs...)
        obj = new(meta(Properties_MultiScalarMolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_MultiScalarMolecularProperty
const __meta_Properties_MultiScalarMolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_MultiScalarMolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_MultiScalarMolecularProperty)
            __meta_Properties_MultiScalarMolecularProperty[] = target = ProtoMeta(Properties_MultiScalarMolecularProperty)
            pack = Symbol[:value]
            allflds = Pair{Symbol,Union{Type,String}}[:value => Base.Vector{Float64}]
            meta(target, Properties_MultiScalarMolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_MultiScalarMolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_MultiScalarMolecularProperty, name::Symbol)
    if name === :value
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_AtomicMolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_AtomicMolecularProperty(; kwargs...)
        obj = new(meta(Properties_AtomicMolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_AtomicMolecularProperty
const __meta_Properties_AtomicMolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_AtomicMolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_AtomicMolecularProperty)
            __meta_Properties_AtomicMolecularProperty[] = target = ProtoMeta(Properties_AtomicMolecularProperty)
            pack = Symbol[:values,:precision]
            allflds = Pair{Symbol,Union{Type,String}}[:values => Base.Vector{Float64}, :precision => Base.Vector{Float64}]
            meta(target, Properties_AtomicMolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_AtomicMolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_AtomicMolecularProperty, name::Symbol)
    if name === :values
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    elseif name === :precision
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_Vector3DMolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_Vector3DMolecularProperty(; kwargs...)
        obj = new(meta(Properties_Vector3DMolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_Vector3DMolecularProperty
const __meta_Properties_Vector3DMolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_Vector3DMolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_Vector3DMolecularProperty)
            __meta_Properties_Vector3DMolecularProperty[] = target = ProtoMeta(Properties_Vector3DMolecularProperty)
            allflds = Pair{Symbol,Union{Type,String}}[:x => Float64, :y => Float64, :z => Float64]
            meta(target, Properties_Vector3DMolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_Vector3DMolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_Vector3DMolecularProperty, name::Symbol)
    if name === :x
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :y
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :z
        return (obj.__protobuf_jl_internal_values[name])::Float64
    else
        getfield(obj, name)
    end
end

mutable struct Properties_VectorFrequentialProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_VectorFrequentialProperty(; kwargs...)
        obj = new(meta(Properties_VectorFrequentialProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_VectorFrequentialProperty
const __meta_Properties_VectorFrequentialProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_VectorFrequentialProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_VectorFrequentialProperty)
            __meta_Properties_VectorFrequentialProperty[] = target = ProtoMeta(Properties_VectorFrequentialProperty)
            allflds = Pair{Symbol,Union{Type,String}}[:magnitude => Float64, :harmonic_intensity => Float64, :normal_mode => Float64]
            meta(target, Properties_VectorFrequentialProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_VectorFrequentialProperty[]
    end
end
function Base.getproperty(obj::Properties_VectorFrequentialProperty, name::Symbol)
    if name === :magnitude
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :harmonic_intensity
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :normal_mode
        return (obj.__protobuf_jl_internal_values[name])::Float64
    else
        getfield(obj, name)
    end
end

mutable struct Properties_AtomicVectorFrequentialProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_AtomicVectorFrequentialProperty(; kwargs...)
        obj = new(meta(Properties_AtomicVectorFrequentialProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_AtomicVectorFrequentialProperty
const __meta_Properties_AtomicVectorFrequentialProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_AtomicVectorFrequentialProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_AtomicVectorFrequentialProperty)
            __meta_Properties_AtomicVectorFrequentialProperty[] = target = ProtoMeta(Properties_AtomicVectorFrequentialProperty)
            allflds = Pair{Symbol,Union{Type,String}}[:frequencies => Base.Vector{Properties_VectorFrequentialProperty}]
            meta(target, Properties_AtomicVectorFrequentialProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_AtomicVectorFrequentialProperty[]
    end
end
function Base.getproperty(obj::Properties_AtomicVectorFrequentialProperty, name::Symbol)
    if name === :frequencies
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Properties_VectorFrequentialProperty}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_Rank2MolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_Rank2MolecularProperty(; kwargs...)
        obj = new(meta(Properties_Rank2MolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_Rank2MolecularProperty
const __meta_Properties_Rank2MolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_Rank2MolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_Rank2MolecularProperty)
            __meta_Properties_Rank2MolecularProperty[] = target = ProtoMeta(Properties_Rank2MolecularProperty)
            pack = Symbol[:matrix_values]
            allflds = Pair{Symbol,Union{Type,String}}[:matrix_values => Base.Vector{Float64}]
            meta(target, Properties_Rank2MolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_Rank2MolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_Rank2MolecularProperty, name::Symbol)
    if name === :matrix_values
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_Rank3MolecularProperty <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_Rank3MolecularProperty(; kwargs...)
        obj = new(meta(Properties_Rank3MolecularProperty), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_Rank3MolecularProperty
const __meta_Properties_Rank3MolecularProperty = Ref{ProtoMeta}()
function meta(::Type{Properties_Rank3MolecularProperty})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_Rank3MolecularProperty)
            __meta_Properties_Rank3MolecularProperty[] = target = ProtoMeta(Properties_Rank3MolecularProperty)
            pack = Symbol[:tensor_values]
            allflds = Pair{Symbol,Union{Type,String}}[:tensor_values => Base.Vector{Float64}]
            meta(target, Properties_Rank3MolecularProperty, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_Rank3MolecularProperty[]
    end
end
function Base.getproperty(obj::Properties_Rank3MolecularProperty, name::Symbol)
    if name === :tensor_values
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Float64}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_NormalMode_AtomicDisplacement <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_NormalMode_AtomicDisplacement(; kwargs...)
        obj = new(meta(Properties_NormalMode_AtomicDisplacement), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_NormalMode_AtomicDisplacement
const __meta_Properties_NormalMode_AtomicDisplacement = Ref{ProtoMeta}()
function meta(::Type{Properties_NormalMode_AtomicDisplacement})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_NormalMode_AtomicDisplacement)
            __meta_Properties_NormalMode_AtomicDisplacement[] = target = ProtoMeta(Properties_NormalMode_AtomicDisplacement)
            allflds = Pair{Symbol,Union{Type,String}}[:x => Float64, :y => Float64, :z => Float64]
            meta(target, Properties_NormalMode_AtomicDisplacement, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_NormalMode_AtomicDisplacement[]
    end
end
function Base.getproperty(obj::Properties_NormalMode_AtomicDisplacement, name::Symbol)
    if name === :x
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :y
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :z
        return (obj.__protobuf_jl_internal_values[name])::Float64
    else
        getfield(obj, name)
    end
end

mutable struct Properties_NormalMode <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_NormalMode(; kwargs...)
        obj = new(meta(Properties_NormalMode), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_NormalMode
const __meta_Properties_NormalMode = Ref{ProtoMeta}()
function meta(::Type{Properties_NormalMode})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_NormalMode)
            __meta_Properties_NormalMode[] = target = ProtoMeta(Properties_NormalMode)
            allflds = Pair{Symbol,Union{Type,String}}[:displacements => Base.Vector{Properties_NormalMode_AtomicDisplacement}]
            meta(target, Properties_NormalMode, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_NormalMode[]
    end
end
function Base.getproperty(obj::Properties_NormalMode, name::Symbol)
    if name === :displacements
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Properties_NormalMode_AtomicDisplacement}
    else
        getfield(obj, name)
    end
end

mutable struct Properties_CalculationStatistics <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_CalculationStatistics(; kwargs...)
        obj = new(meta(Properties_CalculationStatistics), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_CalculationStatistics
const __meta_Properties_CalculationStatistics = Ref{ProtoMeta}()
function meta(::Type{Properties_CalculationStatistics})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_CalculationStatistics)
            __meta_Properties_CalculationStatistics[] = target = ProtoMeta(Properties_CalculationStatistics)
            allflds = Pair{Symbol,Union{Type,String}}[:computing_location => AbstractString, :timings => AbstractString]
            meta(target, Properties_CalculationStatistics, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_CalculationStatistics[]
    end
end
function Base.getproperty(obj::Properties_CalculationStatistics, name::Symbol)
    if name === :computing_location
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :timings
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct Properties_GaussianSanityCheck <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties_GaussianSanityCheck(; kwargs...)
        obj = new(meta(Properties_GaussianSanityCheck), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties_GaussianSanityCheck
const __meta_Properties_GaussianSanityCheck = Ref{ProtoMeta}()
function meta(::Type{Properties_GaussianSanityCheck})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties_GaussianSanityCheck)
            __meta_Properties_GaussianSanityCheck[] = target = ProtoMeta(Properties_GaussianSanityCheck)
            allflds = Pair{Symbol,Union{Type,String}}[:energy_pbe0_6_311gd_diff => Float32, :max_force => Float32, :max_frequencies_diff => Float32, :mean_frequencies_diff => Float32, :max_intensities_diff => Float32, :mean_intensities_diff => Float32, :energy_hf_6_31gd_diff => Float32, :max_dipole_components_diff => Float32, :max_quadropole_components_diff => Float32, :max_octopole_components_diff => Float32]
            meta(target, Properties_GaussianSanityCheck, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties_GaussianSanityCheck[]
    end
end
function Base.getproperty(obj::Properties_GaussianSanityCheck, name::Symbol)
    if name === :energy_pbe0_6_311gd_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :max_force
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :max_frequencies_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :mean_frequencies_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :max_intensities_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :mean_intensities_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :energy_hf_6_31gd_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :max_dipole_components_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :max_quadropole_components_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    elseif name === :max_octopole_components_diff
        return (obj.__protobuf_jl_internal_values[name])::Float32
    else
        getfield(obj, name)
    end
end

mutable struct Properties <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Properties(; kwargs...)
        obj = new(meta(Properties), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Properties
const __meta_Properties = Ref{ProtoMeta}()
function meta(::Type{Properties})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Properties)
            __meta_Properties[] = target = ProtoMeta(Properties)
            fnum = Int[1,2,4,173,174,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,120,43,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,110,111,112,113,114,115,116,117,118,121,119,172]
            allflds = Pair{Symbol,Union{Type,String}}[:errors => Properties_Errors, :geometry_gradient_norms => Properties_AtomicMolecularProperty, :harmonic_intensities => Properties_MultiScalarMolecularProperty, :harmonic_frequencies => Properties_MultiScalarMolecularProperty, :normal_modes => Base.Vector{Properties_NormalMode}, :single_point_energy_pbe0d3_6_311gd => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_311gd => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_311gd_mrcc => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_311gd_orca => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_311gd_cat => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_311gd_cat_mrcc => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_311gd_cat_orca => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_aug_pc_1 => Properties_ScalarMolecularProperty, :single_point_energy_hf_6_31gd => Properties_ScalarMolecularProperty, :single_point_energy_b3lyp_6_31ppgdp => Properties_ScalarMolecularProperty, :single_point_energy_b3lyp_aug_pcs_1 => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_6_31ppgdp => Properties_ScalarMolecularProperty, :single_point_energy_pbe0_aug_pcs_1 => Properties_ScalarMolecularProperty, :single_point_energy_hf_tzvp => Properties_ScalarMolecularProperty, :single_point_energy_mp2_tzvp => Properties_ScalarMolecularProperty, :single_point_energy_cc2_tzvp => Properties_ScalarMolecularProperty, :single_point_energy_hf_3 => Properties_ScalarMolecularProperty, :single_point_energy_mp2_3 => Properties_ScalarMolecularProperty, :single_point_energy_hf_4 => Properties_ScalarMolecularProperty, :single_point_energy_mp2_4 => Properties_ScalarMolecularProperty, :single_point_energy_hf_34 => Properties_ScalarMolecularProperty, :single_point_energy_mp2_34 => Properties_ScalarMolecularProperty, :single_point_energy_hf_cvtz => Properties_ScalarMolecularProperty, :single_point_energy_mp2ful_cvtz => Properties_ScalarMolecularProperty, :single_point_energy_hf_2sp => Properties_ScalarMolecularProperty, :single_point_energy_mp2_2sp => Properties_ScalarMolecularProperty, :single_point_energy_ccsd_2sp => Properties_ScalarMolecularProperty, :single_point_energy_ccsd_t_2sp => Properties_ScalarMolecularProperty, :single_point_energy_hf_2sd => Properties_ScalarMolecularProperty, :single_point_energy_mp2_2sd => Properties_ScalarMolecularProperty, :single_point_energy_ccsd_2sd => Properties_ScalarMolecularProperty, :single_point_energy_ccsd_t_2sd => Properties_ScalarMolecularProperty, :single_point_energy_hf_3psd => Properties_ScalarMolecularProperty, :single_point_energy_mp2_3psd => Properties_ScalarMolecularProperty, :single_point_energy_ccsd_3psd => Properties_ScalarMolecularProperty, :single_point_energy_atomic_b5 => Properties_ScalarMolecularProperty, :single_point_energy_atomic_b6 => Properties_ScalarMolecularProperty, :single_point_energy_eccsd => Properties_ScalarMolecularProperty, :zpe_unscaled => Properties_ScalarMolecularProperty, :homo_pbe0_6_311gd => Properties_ScalarMolecularProperty, :lumo_pbe0_6_311gd => Properties_ScalarMolecularProperty, :homo_pbe0_aug_pc_1 => Properties_ScalarMolecularProperty, :lumo_pbe0_aug_pc_1 => Properties_ScalarMolecularProperty, :homo_pbe0_6_31ppgdp => Properties_ScalarMolecularProperty, :lumo_pbe0_6_31ppgdp => Properties_ScalarMolecularProperty, :homo_pbe0_aug_pcs_1 => Properties_ScalarMolecularProperty, :lumo_pbe0_aug_pcs_1 => Properties_ScalarMolecularProperty, :homo_b3lyp_6_31ppgdp => Properties_ScalarMolecularProperty, :lumo_b3lyp_6_31ppgdp => Properties_ScalarMolecularProperty, :homo_b3lyp_aug_pcs_1 => Properties_ScalarMolecularProperty, :lumo_b3lyp_aug_pcs_1 => Properties_ScalarMolecularProperty, :homo_hf_6_31gd => Properties_ScalarMolecularProperty, :lumo_hf_6_31gd => Properties_ScalarMolecularProperty, :homo_hf_tzvp => Properties_ScalarMolecularProperty, :lumo_hf_tzvp => Properties_ScalarMolecularProperty, :homo_hf_3 => Properties_ScalarMolecularProperty, :lumo_hf_3 => Properties_ScalarMolecularProperty, :homo_hf_4 => Properties_ScalarMolecularProperty, :lumo_hf_4 => Properties_ScalarMolecularProperty, :homo_hf_cvtz => Properties_ScalarMolecularProperty, :lumo_hf_cvtz => Properties_ScalarMolecularProperty, :excitation_energies_cc2 => Properties_MultiScalarMolecularProperty, :excitation_oscillator_strengths_cc2 => Properties_MultiScalarMolecularProperty, :nmr_isotropic_shielding_pbe0_6_31ppgdp => Properties_AtomicMolecularProperty, :nmr_isotropic_shielding_pbe0_aug_pcs_1 => Properties_AtomicMolecularProperty, :nmr_isotropic_shielding_b3lyp_6_31ppgdp => Properties_AtomicMolecularProperty, :nmr_isotropic_shielding_b3lyp_aug_pcs_1 => Properties_AtomicMolecularProperty, :partial_charges_mulliken_pbe0_aug_pc_1 => Properties_AtomicMolecularProperty, :partial_charges_mulliken_hf_6_31gd => Properties_AtomicMolecularProperty, :partial_charges_loewdin_pbe0_aug_pc_1 => Properties_AtomicMolecularProperty, :partial_charges_loewdin_hf_6_31gd => Properties_AtomicMolecularProperty, :partial_charges_natural_nbo_pbe0_aug_pc_1 => Properties_AtomicMolecularProperty, :partial_charges_natural_nbo_hf_6_31gd => Properties_AtomicMolecularProperty, :partial_charges_paboon_pbe0_aug_pc_1 => Properties_AtomicMolecularProperty, :partial_charges_paboon_hf_6_31gd => Properties_AtomicMolecularProperty, :partial_charges_esp_fit_pbe0_aug_pc_1 => Properties_AtomicMolecularProperty, :partial_charges_esp_fit_hf_6_31gd => Properties_AtomicMolecularProperty, :dipole_dipole_polarizability_pbe0_aug_pc_1 => Properties_Rank2MolecularProperty, :dipole_dipole_polarizability_hf_6_31gd => Properties_Rank2MolecularProperty, :dipole_moment_pbe0_aug_pc_1 => Properties_Vector3DMolecularProperty, :dipole_moment_hf_6_31gd => Properties_Vector3DMolecularProperty, :quadrupole_moment_pbe0_aug_pc_1 => Properties_Rank2MolecularProperty, :quadrupole_moment_hf_6_31gd => Properties_Rank2MolecularProperty, :octopole_moment_pbe0_aug_pc_1 => Properties_Rank3MolecularProperty, :octopole_moment_hf_6_31gd => Properties_Rank3MolecularProperty, :compute_cluster_info => AbstractString, :symmetry_used_in_calculation => Bool, :initial_geometry_energy => Properties_ScalarMolecularProperty, :initial_geometry_gradient_norm => Properties_ScalarMolecularProperty, :optimized_geometry_energy => Properties_ScalarMolecularProperty, :optimized_geometry_gradient_norm => Properties_ScalarMolecularProperty, :rotational_constants => Properties_Vector3DMolecularProperty, :bond_separation_reaction_left => Properties_StringMolecularProperty, :bond_separation_reaction_right => Properties_StringMolecularProperty, :bond_separation_energy_atomic_b5 => Properties_ScalarMolecularProperty, :bond_separation_energy_atomic_b5_um => Properties_ScalarMolecularProperty, :bond_separation_energy_atomic_b5_um_ci => Properties_ScalarMolecularProperty, :bond_separation_energy_atomic_b6 => Properties_ScalarMolecularProperty, :bond_separation_energy_atomic_b6_um => Properties_ScalarMolecularProperty, :bond_separation_energy_atomic_b6_um_ci => Properties_ScalarMolecularProperty, :bond_separation_energy_eccsd => Properties_ScalarMolecularProperty, :bond_separation_energy_eccsd_um => Properties_ScalarMolecularProperty, :bond_separation_energy_eccsd_um_ci => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_atomic_b5 => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_atomic_b5_um => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_atomic_b5_um_ci => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_atomic_b6 => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_atomic_b6_um => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_atomic_b6_um_ci => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_eccsd => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_eccsd_um => Properties_ScalarMolecularProperty, :atomization_energy_excluding_zpe_eccsd_um_ci => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_atomic_b5 => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_atomic_b5_um => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_atomic_b5_um_ci => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_atomic_b6 => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_atomic_b6_um => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_atomic_b6_um_ci => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_eccsd => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_eccsd_um => Properties_ScalarMolecularProperty, :atomization_energy_including_zpe_eccsd_um_ci => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_atomic_b5 => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_atomic_b5_um => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_atomic_b5_um_ci => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_atomic_b6 => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_atomic_b6_um => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_atomic_b6_um_ci => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_eccsd => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_eccsd_um => Properties_ScalarMolecularProperty, :enthalpy_of_formation_0k_eccsd_um_ci => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_atomic_b5 => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_atomic_b5_um => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_atomic_b5_um_ci => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_atomic_b6 => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_atomic_b6_um => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_atomic_b6_um_ci => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_eccsd => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_eccsd_um => Properties_ScalarMolecularProperty, :enthalpy_of_formation_298k_eccsd_um_ci => Properties_ScalarMolecularProperty, :zpe_atomic => Properties_ScalarMolecularProperty, :zpe_atomic_um => Properties_ScalarMolecularProperty, :zpe_atomic_um_ci => Properties_ScalarMolecularProperty, :number_imaginary_frequencies => Int32, :number_of_optimization_runs => Int32, :nuclear_repulsion_energy => Properties_ScalarMolecularProperty, :diagnostics_d1_ccsd_2sp => Properties_ScalarMolecularProperty, :diagnostics_d1_ccsd_2sd => Properties_ScalarMolecularProperty, :diagnostics_d1_ccsd_3psd => Properties_ScalarMolecularProperty, :diagnostics_t1_ccsd_2sp => Properties_ScalarMolecularProperty, :diagnostics_t1_ccsd_2sd => Properties_ScalarMolecularProperty, :diagnostics_t1_ccsd_3psd => Properties_ScalarMolecularProperty, :diagnostics_t1_ccsd_2sp_excess => Properties_ScalarMolecularProperty, :calculation_statistics => Base.Vector{Properties_CalculationStatistics}, :gaussian_sanity_check => Properties_GaussianSanityCheck]
            meta(target, Properties, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Properties[]
    end
end
function Base.getproperty(obj::Properties, name::Symbol)
    if name === :errors
        return (obj.__protobuf_jl_internal_values[name])::Properties_Errors
    elseif name === :geometry_gradient_norms
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :harmonic_intensities
        return (obj.__protobuf_jl_internal_values[name])::Properties_MultiScalarMolecularProperty
    elseif name === :harmonic_frequencies
        return (obj.__protobuf_jl_internal_values[name])::Properties_MultiScalarMolecularProperty
    elseif name === :normal_modes
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Properties_NormalMode}
    elseif name === :single_point_energy_pbe0d3_6_311gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_311gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_311gd_mrcc
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_311gd_orca
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_311gd_cat
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_311gd_cat_mrcc
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_311gd_cat_orca
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_b3lyp_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_b3lyp_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_pbe0_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_tzvp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_tzvp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_cc2_tzvp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_3
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_3
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_4
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_4
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_34
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_34
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_cvtz
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2ful_cvtz
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_2sp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_2sp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_ccsd_2sp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_ccsd_t_2sp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_2sd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_2sd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_ccsd_2sd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_ccsd_t_2sd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_hf_3psd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_mp2_3psd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_ccsd_3psd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_atomic_b5
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_atomic_b6
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :single_point_energy_eccsd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :zpe_unscaled
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_pbe0_6_311gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_pbe0_6_311gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_pbe0_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_pbe0_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_pbe0_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_pbe0_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_b3lyp_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_b3lyp_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_b3lyp_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_b3lyp_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_hf_tzvp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_hf_tzvp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_hf_3
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_hf_3
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_hf_4
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_hf_4
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :homo_hf_cvtz
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :lumo_hf_cvtz
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :excitation_energies_cc2
        return (obj.__protobuf_jl_internal_values[name])::Properties_MultiScalarMolecularProperty
    elseif name === :excitation_oscillator_strengths_cc2
        return (obj.__protobuf_jl_internal_values[name])::Properties_MultiScalarMolecularProperty
    elseif name === :nmr_isotropic_shielding_pbe0_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :nmr_isotropic_shielding_pbe0_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :nmr_isotropic_shielding_b3lyp_6_31ppgdp
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :nmr_isotropic_shielding_b3lyp_aug_pcs_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_mulliken_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_mulliken_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_loewdin_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_loewdin_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_natural_nbo_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_natural_nbo_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_paboon_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_paboon_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_esp_fit_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :partial_charges_esp_fit_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_AtomicMolecularProperty
    elseif name === :dipole_dipole_polarizability_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_Rank2MolecularProperty
    elseif name === :dipole_dipole_polarizability_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_Rank2MolecularProperty
    elseif name === :dipole_moment_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_Vector3DMolecularProperty
    elseif name === :dipole_moment_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_Vector3DMolecularProperty
    elseif name === :quadrupole_moment_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_Rank2MolecularProperty
    elseif name === :quadrupole_moment_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_Rank2MolecularProperty
    elseif name === :octopole_moment_pbe0_aug_pc_1
        return (obj.__protobuf_jl_internal_values[name])::Properties_Rank3MolecularProperty
    elseif name === :octopole_moment_hf_6_31gd
        return (obj.__protobuf_jl_internal_values[name])::Properties_Rank3MolecularProperty
    elseif name === :compute_cluster_info
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :symmetry_used_in_calculation
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :initial_geometry_energy
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :initial_geometry_gradient_norm
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :optimized_geometry_energy
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :optimized_geometry_gradient_norm
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :rotational_constants
        return (obj.__protobuf_jl_internal_values[name])::Properties_Vector3DMolecularProperty
    elseif name === :bond_separation_reaction_left
        return (obj.__protobuf_jl_internal_values[name])::Properties_StringMolecularProperty
    elseif name === :bond_separation_reaction_right
        return (obj.__protobuf_jl_internal_values[name])::Properties_StringMolecularProperty
    elseif name === :bond_separation_energy_atomic_b5
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_atomic_b5_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_atomic_b5_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_atomic_b6
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_atomic_b6_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_atomic_b6_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_eccsd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_eccsd_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :bond_separation_energy_eccsd_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_atomic_b5
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_atomic_b5_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_atomic_b5_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_atomic_b6
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_atomic_b6_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_atomic_b6_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_eccsd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_eccsd_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_excluding_zpe_eccsd_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_atomic_b5
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_atomic_b5_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_atomic_b5_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_atomic_b6
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_atomic_b6_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_atomic_b6_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_eccsd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_eccsd_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :atomization_energy_including_zpe_eccsd_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_atomic_b5
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_atomic_b5_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_atomic_b5_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_atomic_b6
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_atomic_b6_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_atomic_b6_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_eccsd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_eccsd_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_0k_eccsd_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_atomic_b5
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_atomic_b5_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_atomic_b5_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_atomic_b6
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_atomic_b6_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_atomic_b6_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_eccsd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_eccsd_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :enthalpy_of_formation_298k_eccsd_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :zpe_atomic
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :zpe_atomic_um
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :zpe_atomic_um_ci
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :number_imaginary_frequencies
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :number_of_optimization_runs
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :nuclear_repulsion_energy
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_d1_ccsd_2sp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_d1_ccsd_2sd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_d1_ccsd_3psd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_t1_ccsd_2sp
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_t1_ccsd_2sd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_t1_ccsd_3psd
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :diagnostics_t1_ccsd_2sp_excess
        return (obj.__protobuf_jl_internal_values[name])::Properties_ScalarMolecularProperty
    elseif name === :calculation_statistics
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Properties_CalculationStatistics}
    elseif name === :gaussian_sanity_check
        return (obj.__protobuf_jl_internal_values[name])::Properties_GaussianSanityCheck
    else
        getfield(obj, name)
    end
end

const Conformer_FateCategory = (;[
    Symbol("FATE_UNDEFINED") => Int32(0),
    Symbol("FATE_DUPLICATE_SAME_TOPOLOGY") => Int32(1),
    Symbol("FATE_DUPLICATE_DIFFERENT_TOPOLOGY") => Int32(2),
    Symbol("FATE_GEOMETRY_OPTIMIZATION_PROBLEM") => Int32(3),
    Symbol("FATE_DISASSOCIATED") => Int32(4),
    Symbol("FATE_FORCE_CONSTANT_FAILURE") => Int32(5),
    Symbol("FATE_DISCARDED_OTHER") => Int32(6),
    Symbol("FATE_NO_CALCULATION_RESULTS") => Int32(7),
    Symbol("FATE_CALCULATION_WITH_ERROR") => Int32(8),
    Symbol("FATE_SUCCESS") => Int32(9),
]...)

mutable struct Conformer <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function Conformer(; kwargs...)
        obj = new(meta(Conformer), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct Conformer
const __meta_Conformer = Ref{ProtoMeta}()
function meta(::Type{Conformer})
    ProtoBuf.metalock() do
        if !isassigned(__meta_Conformer)
            __meta_Conformer[] = target = ProtoMeta(Conformer)
            pack = Symbol[:duplicate_of]
            allflds = Pair{Symbol,Union{Type,String}}[:conformer_id => Int32, :original_conformer_index => Int32, :initial_geometries => Base.Vector{Geometry}, :optimized_geometry => Geometry, :duplicated_by => Int32, :duplicate_of => Base.Vector{Int32}, :properties => Properties, :bond_topologies => Base.Vector{BondTopology}, :fate => Int32]
            meta(target, Conformer, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_Conformer[]
    end
end
function Base.getproperty(obj::Conformer, name::Symbol)
    if name === :conformer_id
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :original_conformer_index
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :initial_geometries
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Geometry}
    elseif name === :optimized_geometry
        return (obj.__protobuf_jl_internal_values[name])::Geometry
    elseif name === :duplicated_by
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :duplicate_of
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :properties
        return (obj.__protobuf_jl_internal_values[name])::Properties
    elseif name === :bond_topologies
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{BondTopology}
    elseif name === :fate
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct MultipleConformers <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MultipleConformers(; kwargs...)
        obj = new(meta(MultipleConformers), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct MultipleConformers
const __meta_MultipleConformers = Ref{ProtoMeta}()
function meta(::Type{MultipleConformers})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MultipleConformers)
            __meta_MultipleConformers[] = target = ProtoMeta(MultipleConformers)
            allflds = Pair{Symbol,Union{Type,String}}[:conformers => Base.Vector{Conformer}]
            meta(target, MultipleConformers, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_MultipleConformers[]
    end
end
function Base.getproperty(obj::MultipleConformers, name::Symbol)
    if name === :conformers
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Conformer}
    else
        getfield(obj, name)
    end
end

mutable struct BondTopologySummary <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function BondTopologySummary(; kwargs...)
        obj = new(meta(BondTopologySummary), Dict{Symbol,Any}(), Set{Symbol}())
        values = obj.__protobuf_jl_internal_values
        symdict = obj.__protobuf_jl_internal_meta.symdict
        for nv in kwargs
            fldname, fldval = nv
            fldtype = symdict[fldname].jtyp
            (fldname in keys(symdict)) || error(string(typeof(obj), " has no field with name ", fldname))
            values[fldname] = isa(fldval, fldtype) ? fldval : convert(fldtype, fldval)
        end
        obj
    end
end # mutable struct BondTopologySummary
const __meta_BondTopologySummary = Ref{ProtoMeta}()
function meta(::Type{BondTopologySummary})
    ProtoBuf.metalock() do
        if !isassigned(__meta_BondTopologySummary)
            __meta_BondTopologySummary[] = target = ProtoMeta(BondTopologySummary)
            allflds = Pair{Symbol,Union{Type,String}}[:bond_topology => BondTopology, :count_attempted_conformers => Int32, :count_duplicates_same_topology => Int32, :count_duplicates_different_topology => Int32, :count_failed_geometry_optimization => Int32, :count_kept_geometry => Int32, :count_missing_calculation => Int32, :count_calculation_with_error => Int32, :count_calculation_success => Int32, :count_detected_match_with_error => Int32, :count_detected_match_success => Int32]
            meta(target, BondTopologySummary, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_BondTopologySummary[]
    end
end
function Base.getproperty(obj::BondTopologySummary, name::Symbol)
    if name === :bond_topology
        return (obj.__protobuf_jl_internal_values[name])::BondTopology
    elseif name === :count_attempted_conformers
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_duplicates_same_topology
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_duplicates_different_topology
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_failed_geometry_optimization
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_kept_geometry
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_missing_calculation
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_calculation_with_error
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_calculation_success
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_detected_match_with_error
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :count_detected_match_success
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

export AvailabilityEnum, BondTopology_AtomType, BondTopology_BondType, BondTopology_Bond, BondTopology, TopologyMatches, Geometry_AtomPos, Geometry, Properties_Errors, Properties_StringMolecularProperty, Properties_ScalarMolecularProperty, Properties_MultiScalarMolecularProperty, Properties_AtomicMolecularProperty, Properties_Vector3DMolecularProperty, Properties_VectorFrequentialProperty, Properties_AtomicVectorFrequentialProperty, Properties_Rank2MolecularProperty, Properties_Rank3MolecularProperty, Properties_NormalMode_AtomicDisplacement, Properties_NormalMode, Properties_CalculationStatistics, Properties_GaussianSanityCheck, Properties, Conformer_FateCategory, Conformer, MultipleConformers, BondTopologySummary
