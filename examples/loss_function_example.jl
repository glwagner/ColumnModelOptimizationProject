using 
    OceanTurb,
    ColumnModelOptimizationProject, 
    ColumnModelOptimizationProject.ModularKPPOptimization,
    PyPlot

using ColumnModelOptimizationProject: evaluate_error_time_series! 
    
datapath = "stress_driven_Nsq5.0e-06_Qu_1.0e-03_Nh128_Nz128_averages.jld2"
#datapath = "stress_driven_Nsq1.0e-05_Qu_1.0e-03_Nh128_Nz128_averages.jld2"

              data = ColumnData(datapath)
             model = ModularKPPOptimization.ColumnModel(data, 5minute, N=64) 
default_parameters = DefaultFreeParameters(model, WindMixingParameters)

loss_function_U = TimeAveragedLossFunction(data, fields=:U)
loss_function_V = TimeAveragedLossFunction(data, fields=:V)
loss_function_T = TimeAveragedLossFunction(data, fields=:T)

evaluate_error_time_series!(loss_function_U, default_parameters, model, data)
evaluate_error_time_series!(loss_function_V, default_parameters, model, data)
evaluate_error_time_series!(loss_function_T, default_parameters, model, data)

close("all")

fig, axs = subplots()

U_label = L"\int \left ( \bar u - U_\mathrm{KPP} \right )^2 \mathrm{d} z"
V_label = L"\int \left ( \bar v - V_\mathrm{KPP} \right )^2 \mathrm{d} z"
T_label = L"\int \left ( \bar \theta - T_\mathrm{KPP} \right )^2 \mathrm{d} z"

plot(loss_function_U.error, label=U_label)
plot(loss_function_V.error, label=V_label)
plot(loss_function_T.error, label=T_label)

xlabel("Data index")
ylabel("Error")
legend()

visualize_realizations(model, data, [1, 101, 401], default_parameters)
