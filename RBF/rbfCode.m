clear all
clc
%% II. ѵ����/���Լ�����
%%
% 1. ��������
load('data.mat');
startIndex=200;
cycle=200;
%%
P_train = P_train';
T_train = T_train';
P_test = P_test';
T_test = T_test';
N = size(P_test,2);
Record=zeros(cycle-startIndex+1,1);
row=1;
%% III. RBF�����紴�����������
%%
% 1. ��������
for i=startIndex:cycle
fprintf("��%d��",i);
net = newrbe(P_train,T_train,i);

%%
% 2. �������
T_sim = sim(net,P_test);
%% IV. ��������
%%
% 1. ������error
errorRelTest = abs(T_sim - T_test)./T_test;
errorTest = T_test-T_sim ;
mseTest=mse(errorTest);
%%
% 2. ����ϵ��R^2
R2 = (N * sum(T_sim .* T_test) - sum(T_sim) * sum(T_test))^2 / ((N * sum((T_sim).^2) - (sum(T_sim))^2) * (N * sum((T_test).^2) - (sum(T_test))^2));
Record(row,1)=R2;
row=row+1;
end
genFunction(net,'rbfForBs');