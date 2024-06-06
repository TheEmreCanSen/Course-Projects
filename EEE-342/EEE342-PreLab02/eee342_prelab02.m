%q1
w = logspace(-1,2,100);
for k = 1:100
    s = 1i * w(k);
    G(k) = 20 / (0.5*s+1);
end
subplot(2,1,1)
semilogx(w,20*log10(abs(G)));
grid on
ylabel('Amplitude (dB)');
xlabel('Angular frequency (ω)');
title('Amplitude vs. Angular Frequency');
subplot(2,1,2)
ylabel('Phase(ϕ)');
xlabel('Angular frequency (ω)');
title('Phase vs. Angular Frequency');
semilogx(w,angle(G)*180/pi)
grid on

%q2

t = 0:0.01:100;
Amplitude = zeros(10, 1);
test= zeros(10, 1);
Phase = zeros(10, 1);
cosine_freq = logspace(-1,2,10);

for n=1:10
    x = cos(t*cosine_freq(n)); 
    y=(cos(-atan(0.5*cosine_freq(n)/1)+cosine_freq(n).*t)*20/sqrt((0.5*cosine_freq(n))^2+1^2));
    lap_x=fft(x);
    [x_max,max_x_loc] = max(abs(lap_x));
    lap_y= fft(y);
    [y_max,max_y_loc] = max(abs(lap_y));
    Phase(n)=angle(lap_y(max_y_loc))-angle(lap_x(max_x_loc));
    Amplitude(n)=y_max/x_max;
    test(n)=abs(y)/abs(x);
end

figure(2)

subplot(2,1,1)
semilogx(w,20*log10(abs(G)));
hold on;
semilogx(cosine_freq,20*log10(Amplitude),'x');
ylabel('Amplitude (dB)');
xlabel('Angular frequency (ω)');
title('Amplitude vs. Angular Frequency');
legend('First Question Plot',"Second Question Plot");
grid on
hold off;

subplot(2,1,2)
semilogx(w,angle(G)*180/pi);
hold on;
semilogx(cosine_freq,Phase*180/pi,'x');
ylabel('Phase(ϕ)');
xlabel('Angular frequency (ω)');
title('Phase vs. Angular Frequency');
legend('First Question Plot',"Second Question Plot");
grid on;
hold off;



