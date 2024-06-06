%{
tic
x=[-5, 1.2, 1/2, 3];
toc

tic
for i=40
    x=nchoosek(50,i);
end;
time=toc
%}
x=zeros(1)
tic
for i=40
    x(i)=nchoosek(50,i);
end;
time=toc

%{
figure 
t=linspace(0,1,101);
x=sin(pi*t+pi/3);
plot(t,x,'b');
title('0.01 interval')

hold on;
t=[0:0.025:1];
x=sin(pi*t+pi/3);
plot(t,x,'g');
title('0.025 interval')

hold on;
t=[0:0.05:1];
x=sin(pi*t+pi/3);
plot(t,x,'m');
title('0.05 interval')
hold on;

t=[0:0.2:1];
x=sin(pi*t+pi/3);
plot(t,x,'m');
title('Combined cosine')
hold on;


t=[0:1/8192:1];
f=783;
x1= cos(2*pi*f*t);

figure
plot(t,x1);
ylabel('cos(2πf0t)');
xlabel('t');
sound(x1)

t=[0:1/8192:1];
a=16;
f=880;
x2= (exp(-a*t)).*cos(2*pi*f*t);

figure
plot(t,x2);
ylabel('e^-at*cos(2πf0t)');
xlabel('t');
sound(x2)


t=[0:1/8192:1];
f0=440;
f1=8;
x3= (cos(2*pi*f0*t)).*cos(2*pi*f1*t);

figure
plot(t,x3);
title('f1=8')
ylabel('cos(2πf1t)cos(2πf0t)');
xlabel('t');
sound(x3)


t=[0:1/8192:1];
a=1870;
x4=cos(pi*a*t.^2);
sound(x4)

t=[0:1/8192:2];
x5=cos(2*pi*(-250*t.^2+800*t+4000));
sound(x5)


t=[0:1/8192:1];
a=1870;
ps=pi;
x6=1/2*cos(2*pi*a*t+ps);
sound(x6)
%}




