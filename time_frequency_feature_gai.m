function [fea] = time_frequency_feature_gai(x,fs)  
%本函数用来计算样本信号的时域与频域特征，输入x为1列信号序列，fs为采样频率,t为样本需要的信号长度时间，s秒，m为想要的样本个数
[m,n] =size(x);

for i=1:m
    time_fea=timeDomainFeatures(x(i,:)); %进行时域特征提取
    freq_fea=frequencyDomainFeatures(x(i,:)',fs); %进行频域特征提取
    time_fea = struct2cell(time_fea);%struct转cell
    time_fea = cell2mat(time_fea);%cell转mat
    freq_fea = struct2cell(freq_fea);%struct转cell
    freq_fea = cell2mat(freq_fea);%cell转mat
    fea(i,:)=[time_fea',freq_fea'];    %组合成特征矩阵，每行对应每个样本的时域，频域特征
    %fea(i,:)=[freq_fea'];
    %fea(i,:)=[time_fea']
end
end