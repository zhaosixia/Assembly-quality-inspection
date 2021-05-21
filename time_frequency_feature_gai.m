function [fea] = time_frequency_feature_gai(x,fs)  
%�������������������źŵ�ʱ����Ƶ������������xΪ1���ź����У�fsΪ����Ƶ��,tΪ������Ҫ���źų���ʱ�䣬s�룬mΪ��Ҫ����������
[m,n] =size(x);

for i=1:m
    time_fea=timeDomainFeatures(x(i,:)); %����ʱ��������ȡ
    freq_fea=frequencyDomainFeatures(x(i,:)',fs); %����Ƶ��������ȡ
    time_fea = struct2cell(time_fea);%structתcell
    time_fea = cell2mat(time_fea);%cellתmat
    freq_fea = struct2cell(freq_fea);%structתcell
    freq_fea = cell2mat(freq_fea);%cellתmat
    fea(i,:)=[time_fea',freq_fea'];    %��ϳ���������ÿ�ж�Ӧÿ��������ʱ��Ƶ������
    %fea(i,:)=[freq_fea'];
    %fea(i,:)=[time_fea']
end
end