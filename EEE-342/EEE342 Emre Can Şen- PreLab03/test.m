% a=87.950748;
% k=10;
% b=0.1668426;
% sys=tf([k*b,k*b],[a,2*a+1,2,0]);
% p=pole(sys);
% c=100*0.1668426;
% g_s=tf([c,2*c],[a,3*a+1,3,0]);
% bode(g_s)
% g1=tf(13.583,[0.191989,1]);
% g2=tf([0.147243,11.7794],[1,15.626,0]);
% g3=tf([-0.005,1],[0.005,1]);
% g=g1*g2*g3;
% bode(g)
% [Gm,Pm,Wcg,Wcp]=margin(g);
% qu=tf([1, -1],[1,3,3,1]);
% bode(qu)

%for i = 1:2:10
%    i
%end

figure(1)
x=[0 0.1 0.2 0.3 0.4 0.5];
y=[17.599831581115723 29.643325090408325 42.39787530899048 57.49822235107422 71.56325674057007 85.5674638748169 ];
stem(x,y)
xlabel("Loss Probability")
ylabel("File Transfer Time(s)")
title("Loss Probability vs. Transfer Time")

figure(2)
x=[0 0.1 0.2 0.3 0.4 0.5];
y=[2286997*8/17.599831581115723 2286997*8/29.643325090408325 2286997*8/42.39787530899048 2286997*8/57.49822235107422 2286997*8/71.56325674057007 2286997*8/85.5674638748169 ];
plot(x,y)
xlabel("Loss Probability")
ylabel("Throughput(bps)")
title("Loss Probability vs. Throughput")

figure(3)
x=[20 40 60 80 100];
y=[31.873268604278564 21.850029230117798 18.381356716156006 16.195859670639038 11.94794511795044];
plot(x,y)
xlabel("Window Size")
ylabel("File Transfer Time(s))")
title("Window Size vs. Transfer Time")

% figure(4)
% x=[20 40 60 80 100];
% %y=[31.873268604278564 21.850029230117798 18.381356716156006 16.195859670639038 11.94794511795044];
% y=[30.884020805358887 22.503368854522705 19.910331964492798 16.71030616760254 15.333240985870361];
% plot(x,y)
% xlabel("Window Size")
% ylabel("Throughput(bps)")
% title("Window Size vs. Throughput")

figure(4)
x=[20 40 60 80 100];
y=[2286997*8/31.873268604278564 2286997*8/21.850029230117798 2286997*8/18.381356716156006 2286997*8/16.195859670639038 2286997*8/11.94794511795044];
plot(x,y)
xlabel("Window Size")
ylabel("Throughput(bps)")
title("Window Size vs. Throughput")