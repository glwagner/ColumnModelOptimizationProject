symbols = ["o", "^", "s", "*", "d", "<"]

fig, axs = subplots()

for (i, opt) in enumerate(optimals)
    N² = buoyancy_frequency(datums[i])
    plot(opt, symbols[i], label=@sprintf("\$ N^2 = %.1e \\, \\mathrm{s^{-2}} \$", N²), alpha=0.6)
end

tick_labels = [ parameter_latex_guide[p] for p in propertynames(optimals[1]) ]
xticks(0:length(tick_labels)-1, tick_labels)

legend()

