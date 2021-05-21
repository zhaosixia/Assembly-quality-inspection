function [ frequencystruct ] = frequencyDomainFeatures(src,fs)
%计算频域统计特征 src输入信号必须是列向量！
%***********************对信号进行FFT变换*******************************
FS=fs;
N=length(src);
n=0:N-1;
freq=n*fs/N; 
f=abs(fft(src,N)*2/N); 

x=f(1:N/2);  %纵坐标    频率幅值
freq=freq(1:N/2)';  %横坐标    频率值

% plot(freq,x);
% title('原始信号频域波形');
% xlabel('频率/hz');
% ylabel('幅值/v');

%********************************计算频域特征值*****************************
frequencystruct.MF=mean(x); %平均频率
frequencystruct.FC=sum(freq.*x)/sum(x);%中心频率
frequencystruct.RMSF=sqrt(sum([freq.^2].*x)/sum(x));%频率均方根
frequencystruct.RVF=sqrt(sum([(freq-frequencystruct.FC).^2].*x)/length(x));%频率标准差
frequencystruct.KURF = sum([(freq-frequencystruct.FC).^4].*x)/(length(x)*(frequencystruct.RVF).^4) %Kurtosis frequency 频率峭度
%第一级边频带幅值和=上下最靠近啮合频率的边频带对应序号的频率幅值    这里随便写
%已知：行星轮个数、转频，可以找到啮合频率，然后得到边频带，即可计算

 %frequencystruct.FSB=x(1000)+x(2000);%第一级边频带幅值和
 %frequencystruct.FSI=frequencystruct.FSB/2;%边频带指数
% frequencystruct.FM0=(max(x)-min(x))/frequencystruct.FSB;%FM0
% frequencystruct.FSLF=frequencystruct.FSB/std(x,1);%边频带等级因子

end



