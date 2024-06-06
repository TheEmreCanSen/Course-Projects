%%%     Part 1    %%%

close all; 
clear all;
rand('seed', sum(100*clock));

x1=-12+12*rand(1,1000000);
x2=2*rand(1,1000000);

y=x1+x2;

figure 
histogram(y)
title("Histogram of 1M Source Samples Y")
xlabel("Values");
ylabel("Sample Count");

sqr_y=0;
for i=1:length(y)
    sqr_y=sqr_y+y(i)^2;
end
sqr_exp_y=sqr_y/length(y);

q_point=linspace(-12,2,5);

for i=1:4
    mid_point(i)=(q_point(i)+q_point(i+1))/2;
end

for i=1:length(y)
   k=1;
   while y(i)>q_point(k+1)
       k=k+1;
   end
   y_quantized_n4(i)=mid_point(k);
end
figure 
histogram(y-y_quantized_n4);
title("Quantization Error Histogram N=4")
xlabel("Quantization Error Value");
ylabel("Error Count");

q_point=linspace(-12,2,33);

for i=1:32
    mid_point(i)=(q_point(i)+q_point(i+1))/2;
end

for i=1:length(y)
   k=1;
   while y(i)>q_point(k+1)
       k=k+1;
   end
   y_quantized_n32(i)=mid_point(k);
end

figure 
histogram(y-y_quantized_n32);
title("Quantization Error Histogram N=32")
xlabel("Quantization Error Value");
ylabel("Error Count");

quantization_error_n4=0;
for i=1:length(y)
quantization_error_n4=quantization_error_n4+(y_quantized_n4(i)-y(i))^2;
end
quantization_error_n4=quantization_error_n4/length(y);

avg_power_qntz_error_n4=10*log10(quantization_error_n4);
sqnr_n4=10.*log10(sqr_exp_y/quantization_error_n4);


quantization_error_n32=0;
for i=1:length(y)
quantization_error_n32=quantization_error_n32+(y_quantized_n32(i)-y(i))^2;
end
quantization_error_n32=quantization_error_n32/length(y);

avg_power_qntz_error_n32=10*log10(quantization_error_n32);
sqnr_n32=10.*log10(sqr_exp_y/quantization_error_n32);


%%%     Lloyd Max    %%%


q_levels_n4=linspace(-12,2,5);
r_levels_n4=zeros(1,4);

d=0;
while d<1000
    for k=1:length(r_levels_n4)
        sum_holder=y(q_levels_n4(k)<y & y<q_levels_n4(k+1));
        r_levels_n4(k)=sum(sum_holder)/(length(sum_holder));
    end
    for m=2:length(q_levels_n4)-1
        q_levels_n4(m)=(r_levels_n4(m-1)+r_levels_n4(m))/2;
    end
    d=d+1;
end

figure
xs=[-12.0000 -8.0558 -4.9970 -1.9395];
stairs(xs, r_levels_n4)
title("Lloyd-Max Quantization Regions & Reconstruction Points")
xlabel("Quantization Region");
ylabel("Reconstruction Points");

for h=1:4
    for t=1:1000000
        if(y(t)<q_levels_n4(h+1) && y(t)>=q_levels_n4(h))
            Y_reconstructed_n4(t)=r_levels_n4(h);
        end
    end
end

figure 
histogram(y-Y_reconstructed_n4);
title("Quantization Error Histogram N=4(Lloyd-Max)")
xlabel("Quantization Error Value");
ylabel("Error Count");


quantization_error_llyod_n4=0;
for i=1:length(y)
    quantization_error_llyod_n4=quantization_error_llyod_n4+(Y_reconstructed_n4(i)-y(i))^2;
end
quantization_error_llyod_n4=quantization_error_llyod_n4/length(y);

avg_power_llyod_error_n4=10*log10(quantization_error_llyod_n4);
sqnr_n4_llyod=10.*log10(sqr_exp_y/quantization_error_llyod_n4);

q_levels_n32=linspace(-12,2,33);
r_levels_n32=zeros(1,32);

d=0;
while d<1000
    for k=1:length(r_levels_n32)
        sum_holder=y(q_levels_n32(k)<y & y<q_levels_n32(k+1));
        r_levels_n32(k)=sum(sum_holder)/(length(sum_holder));
    end
    for m=2:length(q_levels_n32)-1
        q_levels_n32(m)=(r_levels_n32(m-1)+r_levels_n32(m))/2;
    end
    d=d+1;
end


for h=1:32
    for t=1:1000000
        if(y(t)<q_levels_n32(h+1) && y(t)>=q_levels_n32(h))
            Y_reconstructed_n32(t)=r_levels_n32(h);
        end
    end
end

figure 
histogram(y-Y_reconstructed_n32);
title("Quantization Error Histogram N=32(Lloyd-Max)")
xlabel("Quantization Error Value");
ylabel("Error Count");


quantization_error_llyod_n32=0;
for i=1:length(y)
    quantization_error_llyod_n32=quantization_error_llyod_n32+(Y_reconstructed_n32(i)-y(i))^2;
end
quantization_error_llyod_n32=quantization_error_llyod_n32/length(y);

avg_power_llyod_error_n32=10*log10(quantization_error_llyod_n32);
sqnr_n32_llyod=10.*log10(sqr_exp_y/quantization_error_llyod_n32);

%% 
%%%     Part 2    %%%
og_text=fileread("telecom.txt");
og_text=string(og_text);
og_text=lower(og_text);
og_text=append(og_text," #");
og_text=char(og_text);
text_2=og_text;

alphabet=["#" "a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y" "z" " "];   % It's the alphabet that contains the char used in the string
indexes=zeros(1,28);
for l=0:27
   indexes(l+1)=l; 
end
encoded_output_bi="";
lz_dict=dictionary(alphabet,indexes);
character_count=dictionary(alphabet,zeros(1,28));
text_counter=28;
final_text="";
bit_total=0;

for z=1:length(og_text)
    character_count(og_text(z))=character_count(og_text(z))+1;
end

index_length=28;
l=1;
while length(og_text)>1
    i=1;
    text_parse=og_text(1:i);
    while isKey(lz_dict,text_parse)
        i=i+1;
        text_parse=og_text(1:i);
        encoded_output_dec(l)=lz_dict(og_text(1:i-1)); 
        current=encoded_output_dec(l);
    end   
    lz_dict(og_text(1:i))=index_length;
    index_length=index_length+1;
    og_text=og_text(i:length(og_text));
    l=l+1;
    digits=ceil(log(numEntries(lz_dict))/log(2));
    binary_output=(dec2bin(current,digits));
    encoded_output_bi=append(encoded_output_bi,binary_output);
    for i=1:length(binary_output)
        bit_total=bit_total+length(char(num2str(binary_output(i))));
    end
end
bit_sum=strlength(encoded_output_bi);


dict_keys=keys(lz_dict);
text_holder=char(encoded_output_bi);

for i=1:length(encoded_output_dec)
    final_text=append(final_text,dict_keys(bin2dec(text_holder(1:ceil(log2(text_counter+1))))+1));
    text_holder=text_holder(ceil(log2(text_counter+1))+1:length(text_holder));
    text_counter=text_counter+1;
end

for i=keys(character_count)
    character_count(i)=character_count(i)/sum(values(character_count));
end

stem(categorical(keys(character_count)), values(character_count));

entropy=0;
for lamo=values(character_count)
    entropy=entropy+lamo.*log2(lamo);
end

entropy=(-1)*sum(entropy);

R_encoded=bit_sum/strlength(text_2);


