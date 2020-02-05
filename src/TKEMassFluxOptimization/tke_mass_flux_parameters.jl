#
# Parameter sets
#

Base.@kwdef mutable struct WindMixingParameters{T} <: FreeParameters{8, T}
     Cᴸᵏ :: T
     Cᴸᵇ :: T
     Cᴸᵟ :: T
      Cᴰ :: T
     Cᴷᵤ :: T
     Cᴷᵩ :: T
     Cᴷₑ :: T
    Cʷu★ :: T
end

Base.similar(p::WindMixingParameters{T}) where T = 
    WindMixingParameters{T}(0, 0, 0, 0, 0, 0, 0, 0)
