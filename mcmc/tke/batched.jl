using ColumnModelOptimizationProject, JLD2

@free_parameters ParametersToOptimize Cᴷu Cᴷe CᴷPr Cᴰ Cᴸʷ Cʷu★ Cᴸᵇ

include("setup.jl")
include("utils.jl")

LESbrary_path = "/Users/gregorywagner/Projects/BoundaryLayerTurbulenceSimulations/idealized/data"

LES_data = (LESbrary["kato, N²: 1e-5"], LESbrary["ekman, N²: 1e-5"])

