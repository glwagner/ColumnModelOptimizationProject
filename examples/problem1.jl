revolutions = 0.15915494

y = [(revolutions*x)%1 for x=1:30]
t = 1:30

using Plots
plot(t,y,label = false)
scatter!([30], [y[30]], label= "(s,t)")


revolutions = 0.15915494

y = [(revolutions*x)% for x=1:30]
t = 1:30

using Plots
plot(t,y)
