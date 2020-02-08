#
# Parameter sets
#

@inline function Base.similar(p::FreeParameters{T, N}) where {T, N}
    basetype = typeof(p).name.wrapper
    expr = Expr(:call, Expr(:curly, :($basetype), T), zeros(N)...)
    return eval(expr)
end

Base.@kwdef mutable struct WindMixingParameters{T} <: FreeParameters{7, T}
     Cᴸᵏ :: T
     Cᴸᵇ :: T
      Cᴰ :: T
      Cᴷ :: T
    Cᴾʳᵩ :: T
    Cᴾʳₑ :: T
    Cʷu★ :: T
end

Base.@kwdef mutable struct WindMixingFixedPrandtlParameters{T} <: FreeParameters{6, T}
     Cᴸᵏ :: T
     Cᴸᵇ :: T
      Cᴰ :: T
      Cᴷ :: T
    Cᴾʳₑ :: T
    Cʷu★ :: T
end

tke_parameter_sets = (
    WindMixingParameters,
    WindMixingFixedPrandtlParameters,
)

for parameter_set in tke_parameter_sets
    @eval begin
        @inline Base.similar(p::$parameter_set{T}) where {T} =
            $parameter_set{T}(zeros(length(p))...)
    end
end
        
