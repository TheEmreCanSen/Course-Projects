fid = fopen('C:\Users\Emre\Desktop\output_uniform.txt','rt'); 
C = textscan(fid,'%d'); 
fclose(fid); 
YR = cell2mat(C); 
 
x = -1024:1:1024;
y = zeros([2049,1]);

%%
for i = 1:1:length(YR)
    n = YR(i); 
    y(n + 1024 + 1) = y(n + 1024 + 1) + 1;
end

%%



plot(x,y)

