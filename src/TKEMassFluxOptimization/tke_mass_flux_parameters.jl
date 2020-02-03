#
# Parameter sets
#

Base.@kwdef mutable struct WindMixingParameters{T} <: FreeParameters{7, T}
     CLz :: T
     CLb :: T
     CLΔ :: T
     CDe :: T
    CK_U :: T
    CK_T :: T
    CK_e :: T
end

Base.similar(p::WindMixingParameters{T}) where T = 
    WindMixingParameters{T}(0, 0, 0, 0, 0, 0, 0)
