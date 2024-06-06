figure(1)
hold on
plot(t,y,'b');
plot(filtered_out,'r');
legend("Real Data","Filtered Data")
xlabel("t(sec)")
ylabel("Angular Velocity(ω)")
title("Real vs. Filtered Data")
hold off

average=round(mean(filtered_out.Data(11208:49910)));

figure(2)
hold on
plot(t,y,'b');
plot(out.test_out);
legend("Real Data","Approximated Data")
xlabel("t(sec)")
ylabel("Angular Velocity(ω)")
title("Real Data vs. Approximated Data")
hold off