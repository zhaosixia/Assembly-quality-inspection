function [ frequencystruct ] = frequencyDomainFeatures(src,fs)
%����Ƶ��ͳ������ src�����źű�������������
%***********************���źŽ���FFT�任*******************************
FS=fs;
N=length(src);
n=0:N-1;
freq=n*fs/N; 
f=abs(fft(src,N)*2/N); 

x=f(1:N/2);  %������    Ƶ�ʷ�ֵ
freq=freq(1:N/2)';  %������    Ƶ��ֵ

% plot(freq,x);
% title('ԭʼ�ź�Ƶ����');
% xlabel('Ƶ��/hz');
% ylabel('��ֵ/v');

%********************************����Ƶ������ֵ*****************************
frequencystruct.MF=mean(x); %ƽ��Ƶ��
frequencystruct.FC=sum(freq.*x)/sum(x);%����Ƶ��
frequencystruct.RMSF=sqrt(sum([freq.^2].*x)/sum(x));%Ƶ�ʾ�����
frequencystruct.RVF=sqrt(sum([(freq-frequencystruct.FC).^2].*x)/length(x));%Ƶ�ʱ�׼��
frequencystruct.KURF = sum([(freq-frequencystruct.FC).^4].*x)/(length(x)*(frequencystruct.RVF).^4) %Kurtosis frequency Ƶ���Ͷ�
%��һ����Ƶ����ֵ��=�����������Ƶ�ʵı�Ƶ����Ӧ��ŵ�Ƶ�ʷ�ֵ    �������д
%��֪�������ָ�����תƵ�������ҵ�����Ƶ�ʣ�Ȼ��õ���Ƶ�������ɼ���

 %frequencystruct.FSB=x(1000)+x(2000);%��һ����Ƶ����ֵ��
 %frequencystruct.FSI=frequencystruct.FSB/2;%��Ƶ��ָ��
% frequencystruct.FM0=(max(x)-min(x))/frequencystruct.FSB;%FM0
% frequencystruct.FSLF=frequencystruct.FSB/std(x,1);%��Ƶ���ȼ�����

end



