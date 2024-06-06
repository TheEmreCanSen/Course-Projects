%Question 1-C%
x_t=load('signaldata2.mat');
x_t=struct2array(x_t);
t=0:1/1200:1;
a=1;
k=1;
sin_fk=[];
fk=[30*k-20,30*k-10,30*k];
sum_list_holder=[];
sum_list=[];
max_sum_list=[];

for k=1:10
    fk=[30*k-20,30*k-10,30*k];
    for xd=1:3
        for i=1:length(t)
            sin_fk(a)=sin(2*pi*fk(xd)*i/1200);
            a=a+1;
        end
        temp_total=x_t.*sin_fk;
        sum_list_holder(xd) = sum(temp_total);
        a=1;
    end
    [M,I] = max(abs(sum_list_holder));
    sum_list(k) = fk(I); 
    max_sum_list(k) = M;
    sum_list_holder = [];
end 

ak_list=max_sum_list/600;

xt_check=ak_list.*sin(2*pi*sum_list/1200);
xt_check_sum=sum(xt_check);

%q1-d%
for i = 1:length(x_t)
    noise1= (sqrt(1))*randn(1);
end

for i = 1:length(x_t)
    noise25= (sqrt(25))*randn(1);
end

for i = 1:length(x_t)
    noise100= (sqrt(100))*randn(1);
end

y = x_t + noise1;
figure()
plot(t,y);
hold on;
plot(t,x_t);
title('Original vs. Noisy Signal - Variance = 1')
xlabel('Time')
ylabel('Signal')

y = x_t + noise25;
figure()
plot(t,y);
hold on;
plot(t,x_t);
title('Original vs. Noisy Signal - Variance = 25')
xlabel('Time')
ylabel('Signal')


y = x_t + noise100;
figure()
plot(t,y);
hold on;
plot(t,x_t);
title('Original vs. Noisy Signal - Variance = 100')
xlabel('Time')
ylabel('Signal')

%q1-e%
y_t = x_t + noise1;
sum_list1= [];

for k=1:10
    fk=[30*k-20,30*k-10,30*k];
    for xd=1:3
        for i=1:length(t)
            sin_fk(a)=sin(2*pi*fk(xd)*i/1200);
            a=a+1;
        end
        temp_total=y_t.*sin_fk;
        sum_list_holder(xd) = sum(temp_total);
        a=1;
    end
    [M,I] = max(abs(sum_list_holder));
    sum_list1(k) = fk(I); 
    max_sum_list(k) = M;
    sum_list_holder = [];
end

ak_list_1=max_sum_list/600;

xt_check=ak_list_1.*sin(2*pi*sum_list1/1200);
xt_check_sum_1=sum(xt_check);


y_t = x_t + noise25;

for k=1:10
    fk=[30*k-20,30*k-10,30*k];
    for xd=1:3
        for i=1:length(t)
            sin_fk(a)=sin(2*pi*fk(xd)*i/1200);
            a=a+1;
        end
        temp_total=y_t.*sin_fk;
        sum_list_holder(xd) = sum(temp_total);
        a=1;
    end
    [M,I] = max(abs(sum_list_holder));
    sum_list25(k) = fk(I); 
    max_sum_list(k) = M;
    sum_list_holder = [];
end

ak_list_25=max_sum_list/600;

xt_check=ak_list_25.*sin(2*pi*sum_list25/1200);
xt_check_sum_25=sum(xt_check);

y_t = x_t + noise100;
sum_list100 = [];
for k=1:10
    fk=[30*k-20,30*k-10,30*k];
    for xd=1:3
        for i=1:length(t)
            sin_fk(a)=sin(2*pi*fk(xd)*i/1200);
            a=a+1;
        end
        temp_total=y_t.*sin_fk;
        sum_list_holder(xd) = sum(temp_total);
        a=1;
    end
    [M,I] = max(abs(sum_list_holder));
    sum_list100(k) = fk(I); 
    max_sum_list(k) = M;
    sum_list_holder = [];
end

ak_list_100=max_sum_list/600;

xt_check=ak_list_100.*sin(2*pi*sum_list100/1200);
xt_check_sum_100=sum(xt_check);


%%%%%%Q.2%%%%%%

fs = 1000;
period = 0.1;
t = linspace(0,period,100);
map_func = zeros(1, length(t));
bit_count = 5;

for i = 1:length(t)
    if ( 0 <= t(i)) && (t(i)<= period/4)
        map_func(i) = t(i); 
    elseif ( period/4 <= t(i))  &&  (t(i) <= (period)/2)
        map_func(i) = period/2-t(i);
    elseif ((period)/2 <= t(i)) &&  (t(i)<= 3*period/4)
        map_func(i) = t(i)-period/2;
    elseif ((3*period)/4 <= t(i)) &&  (t(i)<=period)
        map_func(i) = period - t(i);
    else
        map_func(i) = 0;
    end
end

t2 = linspace(0,bit_count*period,bit_count*100);
sent = randi([0 1],1,bit_count);

% construct x
x = [];
for k = 1:bit_count
    if (sent(k) == 1)
        x = [x map_func];
    else 
        x = [x -map_func];
    end
end

figure()
plot(t2,x)
title('Reconstruction of x(t)- (5 Bits)')
xlabel('Time')
ylabel('Reconstructed Signal')

figure()
basis = map_func/sqrt(sum(map_func.^2)/fs);
plot(t,basis)
title('Basis Vector')
xlabel('Time')
ylabel('Basis Vector')

sigma_square = [1 0.01 0.0001];
noise_list = zeros(1,3);

for i = 1:length(x)
    noise1(i)= (sqrt(sigma_square(1)))*randn(1);
end

for i = 1:length(x)
    noise2(i)= (sqrt(sigma_square(2)))*randn(1);
end

for i = 1:length(x)
    noise3(i)= (sqrt(sigma_square(3)))*randn(1);
end

noise_list = [noise1(:), noise2(:), noise3(:)];


for i = 1:3
    y = x + noise_list(:,i).';
    subplot(3,1,i);
    plot(t2,y);
    hold on;
    plot(t2,x);
    title(['Variance = ',num2str(sigma_square(i))])
    xlabel('Time')
    ylabel('Signal')
    hold off;
end

%part-e%

fs = 1000;
period = 0.1;
t = linspace(0,period,100);
map_func = zeros(1, length(t));
bit_count = 100000;

for i = 1:length(t)
    if ( 0 <= t(i)) && (t(i)<= period/4)
        map_func(i) = t(i); 
    elseif ( period/4 <= t(i))  &&  (t(i) <= (period)/2)
        map_func(i) = period/2-t(i);
    elseif ((period)/2 <= t(i)) &&  (t(i)<= 3*period/4)
        map_func(i) = t(i)-period/2;
    elseif ((3*period)/4 <= t(i)) &&  (t(i)<=period)
        map_func(i) = period - t(i);
    else
        map_func(i) = 0;
    end
end

t2 = linspace(0,bit_count*period,bit_count*100);
sent = randi([0 1],1,bit_count);

x = [];
for k = 1:bit_count
    if (sent(k) == 1)
        x = [x map_func];
    else 
        x = [x -map_func];
    end
end

energy_binary = sum(map_func.^2);
A = sqrt(sum(map_func.^2));
snr_list = [];
snrdb = [];
snrdb2 = [];
error = [];
Qval= [];
basis = map_func/sqrt(sum(map_func.^2)/fs);
for i= 1:10
    receiver_var= [];
    for k = 1:length(x)
        noise_e(k)= (sqrt(1/(100*i)))*randn(1);
    end
    y = x + noise_e;   
    for a = 1:bit_count
        rk = y(1+100*(a-1):100*(a));
        corr_ml = sum(basis.*rk)/fs;
        if (corr_ml >=  0)
            receiver_var = [receiver_var  1];
        else 
            receiver_var = [receiver_var  0];
        end
    end

    error_holder = sum(abs(receiver_var-sent))/bit_count;
    error = [error error_holder];
    snrdb_pb2 = 10*log10(20*energy_binary/(((1/100*i))));
    num = qfunc(A/sqrt((1/(100*i))));
    Qval = [Qval num];
    snrdb=[snrdb snrdb_pb2];
    snr_holder = snr(x,noise_e);
    snr_list = [snr_list snr_holder];
end

figure()
semilogy(snr_list,error)
title('Probability of Error vs. SNR (ML Rule)')
xlabel('SNR')
ylabel('Probability of Error')


figure()
semilogy(-1*snrdb,error)
title('Probability of Error vs. SNR (ML Rule)')
xlabel('SNR')
ylabel('Probability of Error')


figure()
semilogy(snr_list,Qval)
title('Theoretical Probability of Error vs. SNR (ML Rule)')
xlabel('SNR')
ylabel('Theoretical Probability of Error')


%part-g%

t2 = linspace(0,bit_count*period,bit_count*100);
prior=randsrc(1,bit_count,[1 0;.1 .9]);

x = [];
for k = 1:bit_count
    if (prior(k) == 1)
        x = [x map_func];
    else 
        x = [x -map_func];
    end
end

snr_list = [];
error = [];
basis = map_func/A;

for i= 1:10
    receiver_var= [];
    for k = 1:length(x)
        noise_e(k)= (sqrt(1/(100*i)))*randn(1);
    end
    y = x + noise_e;
    No = 2*(1/(100*i)); 
    for l = 1:bit_count
        rk = y(1+(l-1)*100:(l)*100);
        corr_map = sum(basis.*rk);
    
        if corr_map >=  (-(log(0.1) - log(0.9))*No/(4*A))
            receiver_var = [receiver_var 1];
        else 
            receiver_var = [receiver_var 0];
        end
    end  

    error_holder = sum(abs(receiver_var-prior))/bit_count;
    error = [error error_holder];
   
    snr_holder = snr(x,noise_e);
    snr_list = [snr_list snr_holder];
end

snr_list_map=snr_list;
error_map=error;

for i= 1:10
    receiver_var= [];
    for k = 1:length(x)
        noise_e(k)= (sqrt(1/(100*i)))*randn(1);
    end
    y = x + noise_e;   
    for a = 1:bit_count
        rk = y(1+100*(a-1):100*(a));
        corr_ml = sum(basis.*rk)/fs;
        if (corr_ml >=  0)
            receiver_var = [receiver_var  1];
        else 
            receiver_var = [receiver_var  0];
        end
    end

    error_holder = sum(abs(receiver_var-prior))/bit_count;
    error = [error error_holder];
    snrdb_pb2 = 10*log10(20*energy_binary/(((1/100*i))));
    num = qfunc(A/sqrt((1/(100*i))));
    Qval = [Qval num];
    snrdb=[snrdb snrdb_pb2];
    snr_holder = snr(x,noise_e);
    snr_list = [snr_list snr_holder];
end


figure()
semilogy(snr_list,error)
title('Probability of Error vs. SNR (MAP & ML Rule Comparison)')
xlabel('SNR')
ylabel('Probability of Error')
hold on;
semilogy(snr_list_map,error_map)
legend('ML Rule','MAP Rule')
%% 
alpha = linspace(0.1,0.49,20);

snr_list = [];
error = [];
err_ml = [];
error_ml = [];
basis = map_func/A;

for i= alpha
    
    t2 = linspace(0,bit_count*period,bit_count*100);
    prior=randsrc(1,bit_count,[1 0;i (1-i)]);

    for k = 1:length(x)
        noise_e(k)= (sqrt(1/(100)))*randn(1);
    end
    
    x = [];
    for k = 1:bit_count
        if (prior(k) == 1)
            x = [x map_func];
        else 
            x = [x -map_func];
        end
    end
    receiver_var= [];
    y = x + noise_e;
    No = (1/(100))*2; 
    for l = 1:bit_count
        rk = y(1+(l-1)*100:(l)*100);
        corr_map = sum(basis.*rk);
        if corr_map >=  (-(log(i) - log(1-i))*No/(4*A))
            receiver_var = [receiver_var 1];
        else 
            receiver_var = [receiver_var 0];
        end
    end  
    error_holder = sum(abs(receiver_var-prior))/bit_count;
    error = [error error_holder];
    receiver_var_ml= [];

    for h = 1:bit_count
        r_ml = y(1+(h-1)*100:(h)*100);
        corr_ml = sum(basis.*r_ml)/fs;
        if (corr_ml >=  0)
            receiver_var_ml = [receiver_var_ml  1];
        else 
            receiver_var_ml = [receiver_var_ml  0];
        end
    end
    error_ml = sum(abs(receiver_var_ml-prior))/bit_count;
    err_ml = [err_ml error_ml];
end

figure()
xlabel("\alpha")
ylabel("Probability of Error")
title("Probability of Error vs ML and MAP Rules")
hold on;
plot(alpha,err_ml)
plot(alpha,error)
legend(["ML Rule","MAP Rule"]);