t=[-0.5:0.001:0.5];
t3=[-5:0.001:5];
t4=[-5:0.001:5];
t4b=[-5:0.001:5]-0.7;
A=1+(4).*rand(1,33);
omega=pi.*rand(1,33);
K=37;
K1=4;
K2=10;
K3=18;
K4=52;
K5=109;
K6=34;
T=2;
W=1;
x1=SUMCS(t,A,omega);
x2=FSWave(t3,K,T,W);
x3=FSWave(t4,K1,T,W);
x4=FSWave(t4,K2,T,W);
x5=FSWave(t4,K3,T,W);
x6=FSWave(t4,K4,T,W);
x7=FSWave(t4,K5,T,W);

x8=FSWave(-t4,K6,T,W);

xb=FSWave(t4b,K6,T,W);

xc=FS1Wave(t4,K6,T,W);

xd=FS2Wave(t4,K6,T,W);


figure(1)
subplot(2,2,1);
plot(t,imag(x1))
title('Imaginary vs. t')

subplot(2,2,2);
plot(t,real(x1))
title('Real vs. t')

subplot(2,2,3);
plot(t,abs(x1))
title('Magnitude vs. t')

%((imag(x1)).^2+(real(x1)).^2).^(1/2)

subplot(2,2,4);
plot(t,angle(x1))
title('Angle vs. t')


figure(2)
subplot(2,2,1);
plot(t3,imag(x2))
title('Part 3 - Imaginary vs. t')

subplot(2,2,2);
plot(t3,real(x2)),
title('Part 3 - Real vs. t')

subplot(2,2,3);
plot(t3,abs(x2))
title('Part 3 - Magnitude vs. t')

subplot(2,2,4);
plot(t3,angle(x2))
title('Part 3 - Angle vs. t')

figure(3)
subplot(3,2,1);
plot(t4,real(x3))
title('K=4 - Real vs. t')

subplot(3,2,2);
plot(t4,real(x4))
title('K=10 - Real vs. t')

subplot(3,2,3);
plot(t4,real(x5))
title('K=18 - Real vs. t')

subplot(3,2,4);
plot(t4,real(x6))
title('K=52 - Real vs. t')

subplot(3,2,5);
plot(t4,real(x7))
title('K=109 - Real vs. t')

figure(4)
subplot(2,2,1)
plot(t4,real(x8))
title('Part 4 (a)- Real vs. t')

subplot(2,2,2)
plot(t4,real(xb))
title('Part 4 (b)- Real vs. t')

subplot(2,2,3)
plot(t4,real(xc))
title('Part 4 (c)- Real vs. t')

subplot(2,2,4)
plot(t4,real(xd))
title('Part 4 (d)- Real vs. t')


function xs = SUMCS(t,A,omega)
    xs=zeros(1,length(t));
    for i=1:length(omega)
        xs= xs+A(i)*exp(1j*omega(i)*t);
    end
end

function xt= FSWave(t,K,T,W)
    k0=[-K:K];
    fun=@(x) exp(-1j*2*pi/T*k0*x).*(1-3*x.^2);
    xk=1/T*integral(fun,-W/2, W/2,'ArrayValued',true);
    xt=SUMCS(t,xk,2*pi*k0/T);
end

function xt= FS1Wave(t,K,T,W)
    k0=[-K:K];
    fun=@(x) exp(-1j*2*pi/T*k0*x).*(1-3*x.^2);
    xkz=1/T*integral(fun,-W/2, W/2,'ArrayValued',true);
    xk=(-1j*2*pi/T*k0).*xkz;
    xt=SUMCS(t,xk,2*pi*k0/T);
end

function xt= FS2Wave(t,K,T,W)
    k1=[-K:-1];
    k2=[1:K];
    k1=flip(k1);
    k2=flip(k2);
    k0=[k1 0 k2];
    fun=@(x) exp(-1j*2*pi/T*k0*x).*(1-3*x.^2);
    xk=1/T*integral(fun,-W/2, W/2,'ArrayValued',true);
    xt=SUMCS(t,xk,2*pi*k0/T);
end
