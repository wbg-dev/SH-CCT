clear all
clc
%���ݵ���
load('data.mat');
%ȷ������
global T_train
global P_train
global codeLength;
global inputNum;
global outputNum;
global hiddenLayerSize;
global aa;
%% ��������

P_train=P_train.';
T_train=T_train.';
P_test=P_test.';
T_test=T_test.';

%% ��һ��
% ѵ����
[Pn_train,inputps] = mapminmax(P_train);
Pn_test = mapminmax('apply',P_test,inputps);
% ���Լ�
[Tn_train,outputps] = mapminmax(T_train);
Tn_test = mapminmax('apply',T_test,outputps);
hiddenLayerSize=10;
inputNum = size(P_train,1);
outputNum= size(T_train,1);
codeLength = inputNum*hiddenLayerSize+ hiddenLayerSize*outputNum + inputNum + outputNum;
aa = ones(codeLength,1)*[-1,1];
  
% ��Ⱥ��ģ
popu = 1000; 
% ��ʼ����Ⱥ
initPpp = initializega(popu,aa,'gabpEval',[],[1e-6 1]);  
gen = 1000;  % �Ŵ�����
% ����GAOT�����䣬����Ŀ�꺯������ΪgabpEval
[x,endPop,bPop,trace] = ga(aa,'gabpEval',[],initPpp,[1e-6 1 1],'maxGenTerm',gen,...
                           'normGeomSelect',0.09,'arithXover',2,'nonUnifMutation',[2 gen 3]);

[W1,B1,W2,B2,val] = gadecod(x);

%% ����/ѵ��BP������
net_optimized = newff(Pn_train,Tn_train,hiddenLayerSize);
% ����ѵ������
net_optimized.trainParam.epochs = 1000;
net_optimized.trainParam.show = 10;
net_optimized.trainParam.goal = 1e-7;
net_optimized.trainParam.lr = 0.1;
net_optimized.IW{1,1} = W1;
net_optimized.LW{2,1} = W2;
net_optimized.b{1} = B1;
net_optimized.b{2} = B2;
% �����µ�Ȩֵ����ֵ����ѵ��
net_optimized = train(net_optimized,Pn_train,Tn_train);

%% �������
Tn_sim_optimized = sim(net_optimized,Pn_test);     
% ����һ��
T_sim_optimized = mapminmax('reverse',Tn_sim_optimized,outputps);
error_optimized=T_sim_optimized - T_test;
%% ����Ա�
result_optimized = [T_test' T_sim_optimized'];
% �������
E_optimized = mse(T_sim_optimized - T_test);
% end
% %% δ�Ż���BP������
% % E = zeros(1,100);
% % for i = 1:100
% net = newff(Pn_train,Tn_train,hiddenLayerSize);
% % ����ѵ������
% net.trainParam.epochs = 1000;
% net.trainParam.show = 10;
% net.trainParam.goal = 1e-7;
% net.trainParam.lr = 0.1;
% % �����µ�Ȩֵ����ֵ����ѵ��
% net = train(net,Pn_train,Tn_train);
% 
% %% �������
% Tn_sim = sim(net,Pn_test);    
% % ����һ��
% T_sim = mapminmax('reverse',Tn_sim,outputps);
% 
% %% ����Ա�
% result = [T_test' T_sim'];
% % �������
% E = mse(T_sim - T_test);
% 
% error=T_sim-T_test;
% error_optimized=T_sim_optimized - T_test;
% 
% comparedPredictedValue=[error_optimized' error'];
genFunction(net_optimized,'bsGAModel');
% genFunction(net_optimized,'bsLowerGAModel');