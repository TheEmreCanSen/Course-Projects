%Question 1
w = logspace(-1,2,100);
for k = 1:100
s = 1i * w(k);
G(k) = 8 / (0.14*s+1);
end
figure(1)
subplot(2,1,1)
semilogx(w,20*log10(abs(G)));
grid on
title('Amplitude vs. Angular Frequency');
ylabel('Amplitude (dB)');
xlabel('Angular frequency (W)');
subplot(2,1,2)
semilogx(w,angle(G)*180/pi);
grid on
title('Phase vs. Angular Frequency');
ylabel(['Phase(',char(176),')']);
xlabel('Angular frequency (W)');
%Question 2
t = 0:0.01:100;
Amplitude_detect = zeros(10, 1);
Phase_detect = zeros(10, 1);
cos_freq = logspace(-1,2,10);
for n=1:10
x = cos(t*cos_freq(n)); %create the input cosine signal
y=(cos(-atan(0.14*cos_freq(n)/1)+cos_freq(n).*t)*8/sqrt((0.14*cos_freq(n))^2+1^2));
X_s = fft(x); %take the laplace transform of the input
[x_max,max_x_loc] = max(abs(X_s));
Y_s = fft(y); %take the laplace transform of the output
[y_max,max_y_loc] = max(abs(Y_s));
Phase_detect(n) = angle(Y_s(max_y_loc))-angle(X_s(max_x_loc));
Amplitude_detect(n) = y_max/x_max;
end
figure(2)
subplot(2,1,1)
semilogx(w,20*log10(abs(G)));
hold on;
semilogx(cos_freq,20*log10(Amplitude_detect),'x');
title('Amplitude vs. Angular Frequency');
ylabel('Amplitude (dB)');
xlabel('Angular frequency (W)');
legend('Plot of the first question',"Plot of the second question");
grid on;
hold off;
subplot(2,1,2)
semilogx(w,angle(G)*180/pi);
hold on;
semilogx(cos_freq,Phase_detect*180/pi,'x');
title('Phase vs. Angular Frequency');
ylabel(['Phase(',char(176),')']);
xlabel('Angular frequency (W)');
legend('Plot of the first question',"Plot of the second question");
grid on;
hold off;