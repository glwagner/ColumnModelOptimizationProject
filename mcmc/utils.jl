using Dao

Dao.optimal(result::Dao.AnnealingProblem) = optimal(result.markov_chains[end])

ColumnModelOptimizationProject.visualize_realizations(result::Dao.AnnealingProblem, params; kwargs...) =
       visualize_realizations(result.negative_log_likelihood.model, 
                              result.negative_log_likelihood.data, 
                              result.negative_log_likelihood.loss.targets[[1, end]], params; kwargs...)

function optimum_series(problem, parameter)
    chains = problem.markov_chains
    return [getproperty(optimal(chain).param, parameter) for chain in chains]
end

function optimum_series(problem)
    ParameterType = typeof(problem.markov_chains[1][1].param).name.wrapper
    optimums = [optimum_series(problem, p) for p in fieldnames(ParameterType)]
    return ParameterType(optimums...)
end
    
