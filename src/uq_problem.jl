struct UncertaintyQuantificationProblem{M, D}
    model :: M
    data :: D
    initial_data :: Int
    target_data :: Vector{Int}
end

