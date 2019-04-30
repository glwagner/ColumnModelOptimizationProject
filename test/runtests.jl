using Test, ColumnModelOptimizationProject

function dummy()
    println("Hello, climate.")
    return 1==1
end

@test dummy()
