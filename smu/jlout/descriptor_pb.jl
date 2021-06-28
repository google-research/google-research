# syntax: proto2
using ProtoBuf
import ProtoBuf.meta

mutable struct UninterpretedOption_NamePart <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function UninterpretedOption_NamePart(; kwargs...)
        obj = new(meta(UninterpretedOption_NamePart), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct UninterpretedOption_NamePart
const __meta_UninterpretedOption_NamePart = Ref{ProtoMeta}()
function meta(::Type{UninterpretedOption_NamePart})
    ProtoBuf.metalock() do
        if !isassigned(__meta_UninterpretedOption_NamePart)
            __meta_UninterpretedOption_NamePart[] = target = ProtoMeta(UninterpretedOption_NamePart)
            req = Symbol[:name_part,:is_extension]
            allflds = Pair{Symbol,Union{Type,String}}[:name_part => AbstractString, :is_extension => Bool]
            meta(target, UninterpretedOption_NamePart, allflds, req, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_UninterpretedOption_NamePart[]
    end
end
function Base.getproperty(obj::UninterpretedOption_NamePart, name::Symbol)
    if name === :name_part
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :is_extension
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct UninterpretedOption <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function UninterpretedOption(; kwargs...)
        obj = new(meta(UninterpretedOption), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct UninterpretedOption
const __meta_UninterpretedOption = Ref{ProtoMeta}()
function meta(::Type{UninterpretedOption})
    ProtoBuf.metalock() do
        if !isassigned(__meta_UninterpretedOption)
            __meta_UninterpretedOption[] = target = ProtoMeta(UninterpretedOption)
            fnum = Int[2,3,4,5,6,7,8]
            allflds = Pair{Symbol,Union{Type,String}}[:name => Base.Vector{UninterpretedOption_NamePart}, :identifier_value => AbstractString, :positive_int_value => UInt64, :negative_int_value => Int64, :double_value => Float64, :string_value => Vector{UInt8}, :aggregate_value => AbstractString]
            meta(target, UninterpretedOption, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_UninterpretedOption[]
    end
end
function Base.getproperty(obj::UninterpretedOption, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption_NamePart}
    elseif name === :identifier_value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :positive_int_value
        return (obj.__protobuf_jl_internal_values[name])::UInt64
    elseif name === :negative_int_value
        return (obj.__protobuf_jl_internal_values[name])::Int64
    elseif name === :double_value
        return (obj.__protobuf_jl_internal_values[name])::Float64
    elseif name === :string_value
        return (obj.__protobuf_jl_internal_values[name])::Vector{UInt8}
    elseif name === :aggregate_value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

const FieldOptions_CType = (;[
    Symbol("STRING") => Int32(0),
    Symbol("CORD") => Int32(1),
    Symbol("STRING_PIECE") => Int32(2),
]...)

const FieldOptions_JSType = (;[
    Symbol("JS_NORMAL") => Int32(0),
    Symbol("JS_STRING") => Int32(1),
    Symbol("JS_NUMBER") => Int32(2),
]...)

mutable struct FieldOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FieldOptions(; kwargs...)
        obj = new(meta(FieldOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FieldOptions
const __meta_FieldOptions = Ref{ProtoMeta}()
function meta(::Type{FieldOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FieldOptions)
            __meta_FieldOptions[] = target = ProtoMeta(FieldOptions)
            val = Dict{Symbol,Any}(:ctype => FieldOptions_CType.STRING, :jstype => FieldOptions_JSType.JS_NORMAL, :lazy => false, :deprecated => false, :weak => false)
            fnum = Int[1,2,6,5,3,10,999]
            allflds = Pair{Symbol,Union{Type,String}}[:ctype => Int32, :packed => Bool, :jstype => Int32, :lazy => Bool, :deprecated => Bool, :weak => Bool, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, FieldOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FieldOptions[]
    end
end
function Base.getproperty(obj::FieldOptions, name::Symbol)
    if name === :ctype
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :packed
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :jstype
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :lazy
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :weak
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

mutable struct MessageOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MessageOptions(; kwargs...)
        obj = new(meta(MessageOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct MessageOptions
const __meta_MessageOptions = Ref{ProtoMeta}()
function meta(::Type{MessageOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MessageOptions)
            __meta_MessageOptions[] = target = ProtoMeta(MessageOptions)
            val = Dict{Symbol,Any}(:message_set_wire_format => false, :no_standard_descriptor_accessor => false, :deprecated => false)
            fnum = Int[1,2,3,7,999]
            allflds = Pair{Symbol,Union{Type,String}}[:message_set_wire_format => Bool, :no_standard_descriptor_accessor => Bool, :deprecated => Bool, :map_entry => Bool, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, MessageOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_MessageOptions[]
    end
end
function Base.getproperty(obj::MessageOptions, name::Symbol)
    if name === :message_set_wire_format
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :no_standard_descriptor_accessor
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :map_entry
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

mutable struct EnumOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function EnumOptions(; kwargs...)
        obj = new(meta(EnumOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct EnumOptions
const __meta_EnumOptions = Ref{ProtoMeta}()
function meta(::Type{EnumOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_EnumOptions)
            __meta_EnumOptions[] = target = ProtoMeta(EnumOptions)
            val = Dict{Symbol,Any}(:deprecated => false)
            fnum = Int[2,3,999]
            allflds = Pair{Symbol,Union{Type,String}}[:allow_alias => Bool, :deprecated => Bool, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, EnumOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_EnumOptions[]
    end
end
function Base.getproperty(obj::EnumOptions, name::Symbol)
    if name === :allow_alias
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

mutable struct ExtensionRangeOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ExtensionRangeOptions(; kwargs...)
        obj = new(meta(ExtensionRangeOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ExtensionRangeOptions
const __meta_ExtensionRangeOptions = Ref{ProtoMeta}()
function meta(::Type{ExtensionRangeOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ExtensionRangeOptions)
            __meta_ExtensionRangeOptions[] = target = ProtoMeta(ExtensionRangeOptions)
            fnum = Int[999]
            allflds = Pair{Symbol,Union{Type,String}}[:uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, ExtensionRangeOptions, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ExtensionRangeOptions[]
    end
end
function Base.getproperty(obj::ExtensionRangeOptions, name::Symbol)
    if name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

const MethodOptions_IdempotencyLevel = (;[
    Symbol("IDEMPOTENCY_UNKNOWN") => Int32(0),
    Symbol("NO_SIDE_EFFECTS") => Int32(1),
    Symbol("IDEMPOTENT") => Int32(2),
]...)

mutable struct MethodOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MethodOptions(; kwargs...)
        obj = new(meta(MethodOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct MethodOptions
const __meta_MethodOptions = Ref{ProtoMeta}()
function meta(::Type{MethodOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MethodOptions)
            __meta_MethodOptions[] = target = ProtoMeta(MethodOptions)
            val = Dict{Symbol,Any}(:deprecated => false, :idempotency_level => MethodOptions_IdempotencyLevel.IDEMPOTENCY_UNKNOWN)
            fnum = Int[33,34,999]
            allflds = Pair{Symbol,Union{Type,String}}[:deprecated => Bool, :idempotency_level => Int32, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, MethodOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_MethodOptions[]
    end
end
function Base.getproperty(obj::MethodOptions, name::Symbol)
    if name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :idempotency_level
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

const FileOptions_OptimizeMode = (;[
    Symbol("SPEED") => Int32(1),
    Symbol("CODE_SIZE") => Int32(2),
    Symbol("LITE_RUNTIME") => Int32(3),
]...)

mutable struct FileOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FileOptions(; kwargs...)
        obj = new(meta(FileOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FileOptions
const __meta_FileOptions = Ref{ProtoMeta}()
function meta(::Type{FileOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FileOptions)
            __meta_FileOptions[] = target = ProtoMeta(FileOptions)
            val = Dict{Symbol,Any}(:java_multiple_files => false, :java_string_check_utf8 => false, :optimize_for => FileOptions_OptimizeMode.SPEED, :cc_generic_services => false, :java_generic_services => false, :py_generic_services => false, :php_generic_services => false, :deprecated => false, :cc_enable_arenas => true)
            fnum = Int[1,8,10,20,27,9,11,16,17,18,42,23,31,36,37,39,40,41,44,45,999]
            allflds = Pair{Symbol,Union{Type,String}}[:java_package => AbstractString, :java_outer_classname => AbstractString, :java_multiple_files => Bool, :java_generate_equals_and_hash => Bool, :java_string_check_utf8 => Bool, :optimize_for => Int32, :go_package => AbstractString, :cc_generic_services => Bool, :java_generic_services => Bool, :py_generic_services => Bool, :php_generic_services => Bool, :deprecated => Bool, :cc_enable_arenas => Bool, :objc_class_prefix => AbstractString, :csharp_namespace => AbstractString, :swift_prefix => AbstractString, :php_class_prefix => AbstractString, :php_namespace => AbstractString, :php_metadata_namespace => AbstractString, :ruby_package => AbstractString, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, FileOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FileOptions[]
    end
end
function Base.getproperty(obj::FileOptions, name::Symbol)
    if name === :java_package
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :java_outer_classname
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :java_multiple_files
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :java_generate_equals_and_hash
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :java_string_check_utf8
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :optimize_for
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :go_package
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :cc_generic_services
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :java_generic_services
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :py_generic_services
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :php_generic_services
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :cc_enable_arenas
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :objc_class_prefix
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :csharp_namespace
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :swift_prefix
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :php_class_prefix
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :php_namespace
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :php_metadata_namespace
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :ruby_package
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

mutable struct EnumValueOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function EnumValueOptions(; kwargs...)
        obj = new(meta(EnumValueOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct EnumValueOptions
const __meta_EnumValueOptions = Ref{ProtoMeta}()
function meta(::Type{EnumValueOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_EnumValueOptions)
            __meta_EnumValueOptions[] = target = ProtoMeta(EnumValueOptions)
            val = Dict{Symbol,Any}(:deprecated => false)
            fnum = Int[1,999]
            allflds = Pair{Symbol,Union{Type,String}}[:deprecated => Bool, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, EnumValueOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_EnumValueOptions[]
    end
end
function Base.getproperty(obj::EnumValueOptions, name::Symbol)
    if name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

mutable struct OneofOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function OneofOptions(; kwargs...)
        obj = new(meta(OneofOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct OneofOptions
const __meta_OneofOptions = Ref{ProtoMeta}()
function meta(::Type{OneofOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_OneofOptions)
            __meta_OneofOptions[] = target = ProtoMeta(OneofOptions)
            fnum = Int[999]
            allflds = Pair{Symbol,Union{Type,String}}[:uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, OneofOptions, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_OneofOptions[]
    end
end
function Base.getproperty(obj::OneofOptions, name::Symbol)
    if name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

mutable struct ServiceOptions <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ServiceOptions(; kwargs...)
        obj = new(meta(ServiceOptions), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ServiceOptions
const __meta_ServiceOptions = Ref{ProtoMeta}()
function meta(::Type{ServiceOptions})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ServiceOptions)
            __meta_ServiceOptions[] = target = ProtoMeta(ServiceOptions)
            val = Dict{Symbol,Any}(:deprecated => false)
            fnum = Int[33,999]
            allflds = Pair{Symbol,Union{Type,String}}[:deprecated => Bool, :uninterpreted_option => Base.Vector{UninterpretedOption}]
            meta(target, ServiceOptions, allflds, ProtoBuf.DEF_REQ, fnum, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ServiceOptions[]
    end
end
function Base.getproperty(obj::ServiceOptions, name::Symbol)
    if name === :deprecated
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :uninterpreted_option
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{UninterpretedOption}
    else
        getfield(obj, name)
    end
end

const FieldDescriptorProto_Type = (;[
    Symbol("TYPE_DOUBLE") => Int32(1),
    Symbol("TYPE_FLOAT") => Int32(2),
    Symbol("TYPE_INT64") => Int32(3),
    Symbol("TYPE_UINT64") => Int32(4),
    Symbol("TYPE_INT32") => Int32(5),
    Symbol("TYPE_FIXED64") => Int32(6),
    Symbol("TYPE_FIXED32") => Int32(7),
    Symbol("TYPE_BOOL") => Int32(8),
    Symbol("TYPE_STRING") => Int32(9),
    Symbol("TYPE_GROUP") => Int32(10),
    Symbol("TYPE_MESSAGE") => Int32(11),
    Symbol("TYPE_BYTES") => Int32(12),
    Symbol("TYPE_UINT32") => Int32(13),
    Symbol("TYPE_ENUM") => Int32(14),
    Symbol("TYPE_SFIXED32") => Int32(15),
    Symbol("TYPE_SFIXED64") => Int32(16),
    Symbol("TYPE_SINT32") => Int32(17),
    Symbol("TYPE_SINT64") => Int32(18),
]...)

const FieldDescriptorProto_Label = (;[
    Symbol("LABEL_OPTIONAL") => Int32(1),
    Symbol("LABEL_REQUIRED") => Int32(2),
    Symbol("LABEL_REPEATED") => Int32(3),
]...)

mutable struct FieldDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FieldDescriptorProto(; kwargs...)
        obj = new(meta(FieldDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FieldDescriptorProto
const __meta_FieldDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{FieldDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FieldDescriptorProto)
            __meta_FieldDescriptorProto[] = target = ProtoMeta(FieldDescriptorProto)
            fnum = Int[1,3,4,5,6,2,7,9,10,8,17]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :number => Int32, :label => Int32, :_type => Int32, :type_name => AbstractString, :extendee => AbstractString, :default_value => AbstractString, :oneof_index => Int32, :json_name => AbstractString, :options => FieldOptions, :proto3_optional => Bool]
            meta(target, FieldDescriptorProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FieldDescriptorProto[]
    end
end
function Base.getproperty(obj::FieldDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :number
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :label
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :_type
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :type_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :extendee
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :default_value
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :oneof_index
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :json_name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::FieldOptions
    elseif name === :proto3_optional
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct DescriptorProto_ExtensionRange <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DescriptorProto_ExtensionRange(; kwargs...)
        obj = new(meta(DescriptorProto_ExtensionRange), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DescriptorProto_ExtensionRange
const __meta_DescriptorProto_ExtensionRange = Ref{ProtoMeta}()
function meta(::Type{DescriptorProto_ExtensionRange})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DescriptorProto_ExtensionRange)
            __meta_DescriptorProto_ExtensionRange[] = target = ProtoMeta(DescriptorProto_ExtensionRange)
            allflds = Pair{Symbol,Union{Type,String}}[:start => Int32, :_end => Int32, :options => ExtensionRangeOptions]
            meta(target, DescriptorProto_ExtensionRange, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DescriptorProto_ExtensionRange[]
    end
end
function Base.getproperty(obj::DescriptorProto_ExtensionRange, name::Symbol)
    if name === :start
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :_end
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::ExtensionRangeOptions
    else
        getfield(obj, name)
    end
end

mutable struct MethodDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function MethodDescriptorProto(; kwargs...)
        obj = new(meta(MethodDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct MethodDescriptorProto
const __meta_MethodDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{MethodDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_MethodDescriptorProto)
            __meta_MethodDescriptorProto[] = target = ProtoMeta(MethodDescriptorProto)
            val = Dict{Symbol,Any}(:client_streaming => false, :server_streaming => false)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :input_type => AbstractString, :output_type => AbstractString, :options => MethodOptions, :client_streaming => Bool, :server_streaming => Bool]
            meta(target, MethodDescriptorProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, val, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_MethodDescriptorProto[]
    end
end
function Base.getproperty(obj::MethodDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :input_type
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :output_type
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::MethodOptions
    elseif name === :client_streaming
        return (obj.__protobuf_jl_internal_values[name])::Bool
    elseif name === :server_streaming
        return (obj.__protobuf_jl_internal_values[name])::Bool
    else
        getfield(obj, name)
    end
end

mutable struct EnumValueDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function EnumValueDescriptorProto(; kwargs...)
        obj = new(meta(EnumValueDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct EnumValueDescriptorProto
const __meta_EnumValueDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{EnumValueDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_EnumValueDescriptorProto)
            __meta_EnumValueDescriptorProto[] = target = ProtoMeta(EnumValueDescriptorProto)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :number => Int32, :options => EnumValueOptions]
            meta(target, EnumValueDescriptorProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_EnumValueDescriptorProto[]
    end
end
function Base.getproperty(obj::EnumValueDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :number
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::EnumValueOptions
    else
        getfield(obj, name)
    end
end

mutable struct EnumDescriptorProto_EnumReservedRange <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function EnumDescriptorProto_EnumReservedRange(; kwargs...)
        obj = new(meta(EnumDescriptorProto_EnumReservedRange), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct EnumDescriptorProto_EnumReservedRange
const __meta_EnumDescriptorProto_EnumReservedRange = Ref{ProtoMeta}()
function meta(::Type{EnumDescriptorProto_EnumReservedRange})
    ProtoBuf.metalock() do
        if !isassigned(__meta_EnumDescriptorProto_EnumReservedRange)
            __meta_EnumDescriptorProto_EnumReservedRange[] = target = ProtoMeta(EnumDescriptorProto_EnumReservedRange)
            allflds = Pair{Symbol,Union{Type,String}}[:start => Int32, :_end => Int32]
            meta(target, EnumDescriptorProto_EnumReservedRange, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_EnumDescriptorProto_EnumReservedRange[]
    end
end
function Base.getproperty(obj::EnumDescriptorProto_EnumReservedRange, name::Symbol)
    if name === :start
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :_end
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct EnumDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function EnumDescriptorProto(; kwargs...)
        obj = new(meta(EnumDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct EnumDescriptorProto
const __meta_EnumDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{EnumDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_EnumDescriptorProto)
            __meta_EnumDescriptorProto[] = target = ProtoMeta(EnumDescriptorProto)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :value => Base.Vector{EnumValueDescriptorProto}, :options => EnumOptions, :reserved_range => Base.Vector{EnumDescriptorProto_EnumReservedRange}, :reserved_name => Base.Vector{AbstractString}]
            meta(target, EnumDescriptorProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_EnumDescriptorProto[]
    end
end
function Base.getproperty(obj::EnumDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :value
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{EnumValueDescriptorProto}
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::EnumOptions
    elseif name === :reserved_range
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{EnumDescriptorProto_EnumReservedRange}
    elseif name === :reserved_name
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    else
        getfield(obj, name)
    end
end

mutable struct OneofDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function OneofDescriptorProto(; kwargs...)
        obj = new(meta(OneofDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct OneofDescriptorProto
const __meta_OneofDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{OneofDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_OneofDescriptorProto)
            __meta_OneofDescriptorProto[] = target = ProtoMeta(OneofDescriptorProto)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :options => OneofOptions]
            meta(target, OneofDescriptorProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_OneofDescriptorProto[]
    end
end
function Base.getproperty(obj::OneofDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::OneofOptions
    else
        getfield(obj, name)
    end
end

mutable struct DescriptorProto_ReservedRange <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DescriptorProto_ReservedRange(; kwargs...)
        obj = new(meta(DescriptorProto_ReservedRange), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DescriptorProto_ReservedRange
const __meta_DescriptorProto_ReservedRange = Ref{ProtoMeta}()
function meta(::Type{DescriptorProto_ReservedRange})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DescriptorProto_ReservedRange)
            __meta_DescriptorProto_ReservedRange[] = target = ProtoMeta(DescriptorProto_ReservedRange)
            allflds = Pair{Symbol,Union{Type,String}}[:start => Int32, :_end => Int32]
            meta(target, DescriptorProto_ReservedRange, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DescriptorProto_ReservedRange[]
    end
end
function Base.getproperty(obj::DescriptorProto_ReservedRange, name::Symbol)
    if name === :start
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :_end
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct DescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function DescriptorProto(; kwargs...)
        obj = new(meta(DescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct DescriptorProto
const __meta_DescriptorProto = Ref{ProtoMeta}()
function meta(::Type{DescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_DescriptorProto)
            __meta_DescriptorProto[] = target = ProtoMeta(DescriptorProto)
            fnum = Int[1,2,6,3,4,5,8,7,9,10]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :field => Base.Vector{FieldDescriptorProto}, :extension => Base.Vector{FieldDescriptorProto}, :nested_type => Base.Vector{DescriptorProto}, :enum_type => Base.Vector{EnumDescriptorProto}, :extension_range => Base.Vector{DescriptorProto_ExtensionRange}, :oneof_decl => Base.Vector{OneofDescriptorProto}, :options => MessageOptions, :reserved_range => Base.Vector{DescriptorProto_ReservedRange}, :reserved_name => Base.Vector{AbstractString}]
            meta(target, DescriptorProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_DescriptorProto[]
    end
end
function Base.getproperty(obj::DescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :field
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{FieldDescriptorProto}
    elseif name === :extension
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{FieldDescriptorProto}
    elseif name === :nested_type
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DescriptorProto}
    elseif name === :enum_type
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{EnumDescriptorProto}
    elseif name === :extension_range
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DescriptorProto_ExtensionRange}
    elseif name === :oneof_decl
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{OneofDescriptorProto}
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::MessageOptions
    elseif name === :reserved_range
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DescriptorProto_ReservedRange}
    elseif name === :reserved_name
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    else
        getfield(obj, name)
    end
end

mutable struct ServiceDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function ServiceDescriptorProto(; kwargs...)
        obj = new(meta(ServiceDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct ServiceDescriptorProto
const __meta_ServiceDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{ServiceDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_ServiceDescriptorProto)
            __meta_ServiceDescriptorProto[] = target = ProtoMeta(ServiceDescriptorProto)
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :method => Base.Vector{MethodDescriptorProto}, :options => ServiceOptions]
            meta(target, ServiceDescriptorProto, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_ServiceDescriptorProto[]
    end
end
function Base.getproperty(obj::ServiceDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :method
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{MethodDescriptorProto}
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::ServiceOptions
    else
        getfield(obj, name)
    end
end

mutable struct SourceCodeInfo_Location <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function SourceCodeInfo_Location(; kwargs...)
        obj = new(meta(SourceCodeInfo_Location), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct SourceCodeInfo_Location
const __meta_SourceCodeInfo_Location = Ref{ProtoMeta}()
function meta(::Type{SourceCodeInfo_Location})
    ProtoBuf.metalock() do
        if !isassigned(__meta_SourceCodeInfo_Location)
            __meta_SourceCodeInfo_Location[] = target = ProtoMeta(SourceCodeInfo_Location)
            fnum = Int[1,2,3,4,6]
            pack = Symbol[:path,:span]
            allflds = Pair{Symbol,Union{Type,String}}[:path => Base.Vector{Int32}, :span => Base.Vector{Int32}, :leading_comments => AbstractString, :trailing_comments => AbstractString, :leading_detached_comments => Base.Vector{AbstractString}]
            meta(target, SourceCodeInfo_Location, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_SourceCodeInfo_Location[]
    end
end
function Base.getproperty(obj::SourceCodeInfo_Location, name::Symbol)
    if name === :path
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :span
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :leading_comments
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :trailing_comments
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :leading_detached_comments
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    else
        getfield(obj, name)
    end
end

mutable struct SourceCodeInfo <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function SourceCodeInfo(; kwargs...)
        obj = new(meta(SourceCodeInfo), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct SourceCodeInfo
const __meta_SourceCodeInfo = Ref{ProtoMeta}()
function meta(::Type{SourceCodeInfo})
    ProtoBuf.metalock() do
        if !isassigned(__meta_SourceCodeInfo)
            __meta_SourceCodeInfo[] = target = ProtoMeta(SourceCodeInfo)
            allflds = Pair{Symbol,Union{Type,String}}[:location => Base.Vector{SourceCodeInfo_Location}]
            meta(target, SourceCodeInfo, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_SourceCodeInfo[]
    end
end
function Base.getproperty(obj::SourceCodeInfo, name::Symbol)
    if name === :location
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{SourceCodeInfo_Location}
    else
        getfield(obj, name)
    end
end

mutable struct FileDescriptorProto <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FileDescriptorProto(; kwargs...)
        obj = new(meta(FileDescriptorProto), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FileDescriptorProto
const __meta_FileDescriptorProto = Ref{ProtoMeta}()
function meta(::Type{FileDescriptorProto})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FileDescriptorProto)
            __meta_FileDescriptorProto[] = target = ProtoMeta(FileDescriptorProto)
            fnum = Int[1,2,3,10,11,4,5,6,7,8,9,12]
            allflds = Pair{Symbol,Union{Type,String}}[:name => AbstractString, :package => AbstractString, :dependency => Base.Vector{AbstractString}, :public_dependency => Base.Vector{Int32}, :weak_dependency => Base.Vector{Int32}, :message_type => Base.Vector{DescriptorProto}, :enum_type => Base.Vector{EnumDescriptorProto}, :service => Base.Vector{ServiceDescriptorProto}, :extension => Base.Vector{FieldDescriptorProto}, :options => FileOptions, :source_code_info => SourceCodeInfo, :syntax => AbstractString]
            meta(target, FileDescriptorProto, allflds, ProtoBuf.DEF_REQ, fnum, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FileDescriptorProto[]
    end
end
function Base.getproperty(obj::FileDescriptorProto, name::Symbol)
    if name === :name
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :package
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :dependency
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{AbstractString}
    elseif name === :public_dependency
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :weak_dependency
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :message_type
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{DescriptorProto}
    elseif name === :enum_type
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{EnumDescriptorProto}
    elseif name === :service
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{ServiceDescriptorProto}
    elseif name === :extension
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{FieldDescriptorProto}
    elseif name === :options
        return (obj.__protobuf_jl_internal_values[name])::FileOptions
    elseif name === :source_code_info
        return (obj.__protobuf_jl_internal_values[name])::SourceCodeInfo
    elseif name === :syntax
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    else
        getfield(obj, name)
    end
end

mutable struct FileDescriptorSet <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function FileDescriptorSet(; kwargs...)
        obj = new(meta(FileDescriptorSet), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct FileDescriptorSet
const __meta_FileDescriptorSet = Ref{ProtoMeta}()
function meta(::Type{FileDescriptorSet})
    ProtoBuf.metalock() do
        if !isassigned(__meta_FileDescriptorSet)
            __meta_FileDescriptorSet[] = target = ProtoMeta(FileDescriptorSet)
            allflds = Pair{Symbol,Union{Type,String}}[:file => Base.Vector{FileDescriptorProto}]
            meta(target, FileDescriptorSet, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_FileDescriptorSet[]
    end
end
function Base.getproperty(obj::FileDescriptorSet, name::Symbol)
    if name === :file
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{FileDescriptorProto}
    else
        getfield(obj, name)
    end
end

mutable struct GeneratedCodeInfo_Annotation <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GeneratedCodeInfo_Annotation(; kwargs...)
        obj = new(meta(GeneratedCodeInfo_Annotation), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GeneratedCodeInfo_Annotation
const __meta_GeneratedCodeInfo_Annotation = Ref{ProtoMeta}()
function meta(::Type{GeneratedCodeInfo_Annotation})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GeneratedCodeInfo_Annotation)
            __meta_GeneratedCodeInfo_Annotation[] = target = ProtoMeta(GeneratedCodeInfo_Annotation)
            pack = Symbol[:path]
            allflds = Pair{Symbol,Union{Type,String}}[:path => Base.Vector{Int32}, :source_file => AbstractString, :_begin => Int32, :_end => Int32]
            meta(target, GeneratedCodeInfo_Annotation, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, pack, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GeneratedCodeInfo_Annotation[]
    end
end
function Base.getproperty(obj::GeneratedCodeInfo_Annotation, name::Symbol)
    if name === :path
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{Int32}
    elseif name === :source_file
        return (obj.__protobuf_jl_internal_values[name])::AbstractString
    elseif name === :_begin
        return (obj.__protobuf_jl_internal_values[name])::Int32
    elseif name === :_end
        return (obj.__protobuf_jl_internal_values[name])::Int32
    else
        getfield(obj, name)
    end
end

mutable struct GeneratedCodeInfo <: ProtoType
    __protobuf_jl_internal_meta::ProtoMeta
    __protobuf_jl_internal_values::Dict{Symbol,Any}
    __protobuf_jl_internal_defaultset::Set{Symbol}

    function GeneratedCodeInfo(; kwargs...)
        obj = new(meta(GeneratedCodeInfo), Dict{Symbol,Any}(), Set{Symbol}())
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
end # mutable struct GeneratedCodeInfo
const __meta_GeneratedCodeInfo = Ref{ProtoMeta}()
function meta(::Type{GeneratedCodeInfo})
    ProtoBuf.metalock() do
        if !isassigned(__meta_GeneratedCodeInfo)
            __meta_GeneratedCodeInfo[] = target = ProtoMeta(GeneratedCodeInfo)
            allflds = Pair{Symbol,Union{Type,String}}[:annotation => Base.Vector{GeneratedCodeInfo_Annotation}]
            meta(target, GeneratedCodeInfo, allflds, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
        end
        __meta_GeneratedCodeInfo[]
    end
end
function Base.getproperty(obj::GeneratedCodeInfo, name::Symbol)
    if name === :annotation
        return (obj.__protobuf_jl_internal_values[name])::Base.Vector{GeneratedCodeInfo_Annotation}
    else
        getfield(obj, name)
    end
end

export FileDescriptorSet, FileDescriptorProto, DescriptorProto_ExtensionRange, DescriptorProto_ReservedRange, DescriptorProto, ExtensionRangeOptions, FieldDescriptorProto_Type, FieldDescriptorProto_Label, FieldDescriptorProto, OneofDescriptorProto, EnumDescriptorProto_EnumReservedRange, EnumDescriptorProto, EnumValueDescriptorProto, ServiceDescriptorProto, MethodDescriptorProto, FileOptions_OptimizeMode, FileOptions, MessageOptions, FieldOptions_CType, FieldOptions_JSType, FieldOptions, OneofOptions, EnumOptions, EnumValueOptions, ServiceOptions, MethodOptions_IdempotencyLevel, MethodOptions, UninterpretedOption_NamePart, UninterpretedOption, SourceCodeInfo_Location, SourceCodeInfo, GeneratedCodeInfo_Annotation, GeneratedCodeInfo
