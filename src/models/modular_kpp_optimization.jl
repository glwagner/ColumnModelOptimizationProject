module ModularKPPOptimization

export
    DefaultFreeParameters,
    DefaultStdFreeParameters,
    FreeConvectionParameters,
    ShearUnstableParameters,
    ShearNeutralParameters,
    SensitiveParameters,
    BasicParameters,

    simple_flux_model

using
    OceanTurb,
    StaticArrays,
    PyPlot,
    JLD2

using ..ColumnModelOptimizationProject

import Base: similar
import ColumnModelOptimizationProject: ColumnModel, set!

latexparams = Dict(
      :CRi => L"C^\mathrm{Ri}",
      :CKE => L"C^\mathcal{E}",
      :CNL => L"C^{NL}",
      :Cτ  => L"C^\tau",
    :Cstab => L"C^\mathrm{stab}",
    :Cunst => L"C^\mathrm{unst}",
     :Cb_U => L"C^b_U",
     :Cb_T => L"C^b_T",
     :Cd_U => L"C^d_U",
     :Cd_T => L"C^d_T"
)

include("modular_kpp_utils.jl")

#
# Basic functionality
#
function set!(cm::ColumnModel{<:ModularKPP.Model}, freeparams::FreeParameters{N, T}) where {N, T}

    paramnames, paramtypes = get_free_parameters(cm)
    paramdicts = Dict( ( ptypename, Dict{Symbol, T}() ) for ptypename in keys(paramtypes))

    # Filter freeparams into their appropriate category
    for pname in propertynames(freeparams)
        for ptypename in keys(paramtypes)
            pname ∈ paramnames[ptypename] && push!(paramdicts[ptypename],
                                                   Pair(pname, getproperty(freeparams, pname))
                                                  )
        end
    end

    # Set new parameters
    for (ptypename, PType) in paramtypes
        params = PType(; paramdicts[ptypename]...)
        setproperty!(cm.model, ptypename, params)
    end

    return nothing
end

function set!(cm::ColumnModel{<:ModularKPP.Model}, cd::ColumnData, i)
    set!(cm.model.solution.U, cd.U[i])
    set!(cm.model.solution.V, cd.V[i])
    set!(cm.model.solution.T, cd.T[i])
    #set!(cm.model.solution.S, cd.S[i])

    cm.model.clock.time = cd.t[i]
    return nothing
end

#
# Parameter sets
#

Base.@kwdef mutable struct BasicParameters{T} <: FreeParameters{9, T}
      CRi :: T
      CSL :: T
      CKE :: T
      CNL :: T
      Cτ  :: T
    Cstab :: T
    Cunst :: T
     Cb_U :: T
     Cb_T :: T
end

Base.similar(p::BasicParameters{T}) where T = BasicParameters{T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

function DefaultFreeParameters(cm, freeparamtype)
    paramnames, paramtypes = get_free_parameters(cm)

    alldefaults = (ptype() for ptype in values(paramtypes))

    freeparams = []
    for pname in fieldnames(freeparamtype)
        for ptype in alldefaults
            pname ∈ propertynames(ptype) && push!(freeparams, getproperty(ptype, pname))
        end
    end

    eval(Expr(:call, freeparamtype, freeparams...))
end

function DefaultStdFreeParameters(relative_std, freeparamtype)
    allparams = KPP.Parameters()
    param_stds = (relative_std * getproperty(allparams, name) for name in fieldnames(freeparamtype))
    eval(Expr(:call, freeparamtype, param_stds...))
end

end # module
