%*****************************************************************
%load('JL.mat')
fs=2000;%采样频率
Ts=1/fs;%采样周期
t=(0:1999)*Ts;%时间序列
STA=1; %采样起始位置0
[a,b] = size(CDhc);
%---------------------导入实验分析数据--------------------------------------
PFt = [];
for k = 1:a
    a1 = CDhc(k,:);
    x=a1; %a1'为a1的转置
    %----------------------lmdf分解-------------------
    [PF]=lmd(x);
    %figure(1);
    imfn=PF;
    n = size(imfn,1); %size(X,1),返回矩阵X的行数；size(X,2),返回矩阵X的列数；N=size(X,2)，就是把矩阵X的列数赋值给N
    %subplot(n+1,1,1);  % m代表行，n代表列，p代表的这个图形画在第几行、第几列。例如subplot(2,2,[1,2])
    %plot(t,x); %故障信号
    %ylabel('原始信号','fontsize',12,'fontname','宋体');
    for n1=1:n
       %subplot(n+1,1,n1+1); 
        %plot(t,PF(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
       % ylabel(['PF' int2str(n1)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
    end
        %xlabel('时间\itt/s','fontsize',12,'fontname','宋体');
    c = a1- sum(PF);
    %subplot(n+2,1,n+2);
   % plot(t,c);
    %ylabel('U(t)');
   %----------------------计算每个PF分量的样本熵------------------- 
    for i=1:n 
         u = imfn(i,:);  
         
         a = Shang(u, 2, 0.25*std(u));
      
         Ep(i)= a;
     end
     PFt = [PFt;Ep];
end




