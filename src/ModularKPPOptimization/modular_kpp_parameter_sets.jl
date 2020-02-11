#
# Parameter sets
#

@free_parameters BasicParameters CRi CSL CKE CNL Cτ Cstab Cunst Cb_U Cb_T

@free_parameters WindMixingParameters CRi CSL Cτ

Base.@kwdef mutable struct WindyConvectionParameters{T} <: FreeParameters{15, T}
      CRi :: T
      CSL :: T
     CKSL :: T
      CKE :: T
      CNL :: T
      Cτ  :: T
    Cunst :: T
     Cb_U :: T
     Cb_T :: T
    Cmτ_T :: T
    Cmτ_U :: T
    Cmb_T :: T
    Cmb_U :: T
     Cd_U :: T
     Cd_T :: T
end

Base.similar(p::WindyConvectionParameters{T}) where T =
    WindyConvectionParameters{T}((0 for i = 1:length(fieldnames(WindyConvectionParameters)))...)

Base.similar(p::WindMixingParameters{T}) where T = WindMixingParameters{T}(0, 0, 0)

Base.@kwdef mutable struct WindMixingAndShapeParameters{T} <: FreeParameters{5, T}
      CRi :: T
      CSL :: T
      Cτ  :: T
      CS0 :: T
      CS1 :: T
end

Base.similar(p::WindMixingAndShapeParameters{T}) where T =
    WindMixingAndShapeParameters{T}(0, 0, 0, 0, 0)

Base.@kwdef mutable struct WindMixingAndExponentialShapeParameters{T} <: FreeParameters{6, T}
      CRi :: T
      CSL :: T
      Cτ  :: T
      CS0 :: T
      CSe :: T
      CSd :: T
end

Base.similar(p::WindMixingAndExponentialShapeParameters{T}) where T =
    WindMixingAndExponentialShapeParameters{T}(0, 0, 0, 0, 0, 0)
