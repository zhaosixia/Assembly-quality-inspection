function [ timestruct ] = timeDomainFeatures( src)
% 计算时域统计特征
% src  N*1维矩阵 是要计算特征的源信号

if nargin>2
     error(message('参数有误，只有只能是一个或2个输入参数'));
end

%////////////////////////////////////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 timestruct.max=max(src);%1.最大值
 timestruct.min=min(src);%2.最小值
 timestruct.peak=max(abs(src)); %3.峰值
 timestruct.p2p=max(src)-min(src);%4.峰峰值
 timestruct.mean=mean(src);%5.均值
 %timestruct.averageAmplitude=mean(abs(src));%6.绝对平均值（平均幅值）
 %timestruct.rootAmplitude=mean(sqrt(abs(src)))^2;%7.方根幅值
 %timestruct.var=var(src,1);%8.方差  有偏
 %timestruct.std=std(src,1);%9.标准差
 %timestruct.rms=sqrt(sum(src.^2)/length(src));%10.有效值（均方根）
 %timestruct.kurtosis=kurtosis(src,1);%11.峭度
 %timestruct.skewness=skewness(src,1);%12.偏度
 %timestruct.shapeFactor=timestruct.rms/timestruct.averageAmplitude;%13.波形因子
 %timestruct.peakingFactor=timestruct.peak/timestruct.rms;%14.峰值因子（波峰因子）
% timestruct.pulseFactor=timestruct.peak/timestruct.averageAmplitude;%15.脉冲因子
% %timestruct.marginFactor=timestruct.peak/timestruct.rootAmplitude;%16.裕度因子
% %timestruct.clearanceFactor=timestruct.peak/timestruct.rms^2;%17.余隙因子


%//////////////////////////////////////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% if nargin==2
%      switch(option)
%          case 'max'
%              disp('max value=');
%              disp(timestruct.max);
%          case 'min'
%              disp('min value=');
%              disp(timestruct.min);           
%      end 
% end 

end


