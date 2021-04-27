## For scenarios involving stresses
@free_parameters(TKEParametersRiDependent,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴷu⁻, Cᴷu⁺,
                 Cᴷc⁻, Cᴷc⁺,
                 Cᴷe⁻, Cᴷe⁺,
                 Cᴰ, Cᴸᵇ, Cʷu★, CʷwΔ)

@free_parameters(TKEParametersRiIndependent,
                 Cᴷu, Cᴷc, Cᴷe,
                 Cᴰ, Cᴸᵇ, Cʷu★, CʷwΔ) #Cᴸʷ

@free_parameters(TKEParametersConvectiveAdjustmentRiDependent,
                  CᴷRiʷ, CᴷRiᶜ,
                  Cᴷu⁻, Cᴷu⁺,
                  Cᴷc⁻, Cᴷc⁺,
                  Cᴷe⁻, Cᴷe⁺,
                  Cᴰ, Cᴸᵇ, Cʷu★, CʷwΔ,
                  Cᴬu, Cᴬc, Cᴬe)

@free_parameters(TKEParametersConvectiveAdjustmentRiIndependent,
                  Cᴷu, Cᴷc, Cᴷe,
                  Cᴰ, Cᴸᵇ, Cʷu★, CʷwΔ,
                  Cᴬu, Cᴬc, Cᴬe) #Cᴸʷ

@free_parameters KPPWindMixingParameters CRi CSL Cτ

@free_parameters KPPWindMixingOrConvectionParameters CRi CSL Cτ Cb_U Cb_T

## For purely convective scenarios
@free_parameters(TKEFreeConvection,
                    CᴷRiʷ, CᴷRiᶜ,
                    Cᴷc⁻, Cᴷc⁺,
                    Cᴷe⁻, Cᴷe⁺,
                    Cᴰ, Cᴸᵇ, CʷwΔ)

@free_parameters(TKEFreeConvectionConvectiveAdjustmentRiDependent,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴷc⁻, Cᴷc⁺,
                 Cᴷe⁻, Cᴷe⁺,
                 Cᴰ, Cᴸᵇ, CʷwΔ,
                 Cᴬu, Cᴬc, Cᴬe)

@free_parameters(TKEFreeConvectionConvectiveAdjustmentRiIndependent,
                  Cᴷc, Cᴷe, Cᴰ, Cᴸᵇ, CʷwΔ,
                  Cᴬu, Cᴬc, Cᴬe)

@free_parameters TKEFreeConvectionRiIndependent Cᴷc Cᴷe Cᴰ Cᴸᵇ CʷwΔ
