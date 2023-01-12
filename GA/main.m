clear all
clc
%数据导入
load('data.mat');
%确定参数
global T_train
global P_train
global codeLength;
global inputNum;
global outputNum;
global hiddenLayerSize;
global aa;
%% 导入数据

P_train=P_train.';
T_train=T_train.';
P_test=P_test.';
T_test=T_test.';

%% 归一化
% 训练集
[Pn_train,inputps] = mapminmax(P_train);
Pn_test = mapminmax('apply',P_test,inputps);
% 测试集
[Tn_train,outputps] = mapminmax(T_train);
Tn_test = mapminmax('apply',T_test,outputps);
hiddenLayerSize=10;
inputNum = size(P_train,1);
outputNum= size(T_train,1);
codeLength = inputNum*hiddenLayerSize+ hiddenLayerSize*outputNum + inputNum + outputNum;
aa = ones(codeLength,1)*[-1,1];
  
% 种群规模
popu = 1000; 
% 初始化种群
initPpp = initializega(popu,aa,'gabpEval',[],[1e-6 1]);  
gen = 1000;  % 遗传代数
% 调用GAOT工具箱，其中目标函数定义为gabpEval
[x,endPop,bPop,trace] = ga(aa,'gabpEval',[],initPpp,[1e-6 1 1],'maxGenTerm',gen,...
                           'normGeomSelect',0.09,'arithXover',2,'nonUnifMutation',[2 gen 3]);

[W1,B1,W2,B2,val] = gadecod(x);

%% 创建/训练BP神经网络
net_optimized = newff(Pn_train,Tn_train,hiddenLayerSize);
% 设置训练参数
net_optimized.trainParam.epochs = 1000;
net_optimized.trainParam.show = 10;
net_optimized.trainParam.goal = 1e-7;
net_optimized.trainParam.lr = 0.1;
net_optimized.IW{1,1} = W1;
net_optimized.LW{2,1} = W2;
net_optimized.b{1} = B1;
net_optimized.b{2} = B2;
% 利用新的权值和阈值进行训练
net_optimized = train(net_optimized,Pn_train,Tn_train);

%% 仿真测试
Tn_sim_optimized = sim(net_optimized,Pn_test);     
% 反归一化
T_sim_optimized = mapminmax('reverse',Tn_sim_optimized,outputps);
error_optimized=T_sim_optimized - T_test;
%% 结果对比
result_optimized = [T_test' T_sim_optimized'];
% 均方误差
E_optimized = mse(T_sim_optimized - T_test);
% end
% %% 未优化的BP神经网络
% % E = zeros(1,100);
% % for i = 1:100
% net = newff(Pn_train,Tn_train,hiddenLayerSize);
% % 设置训练参数
% net.trainParam.epochs = 1000;
% net.trainParam.show = 10;
% net.trainParam.goal = 1e-7;
% net.trainParam.lr = 0.1;
% % 利用新的权值和阈值进行训练
% net = train(net,Pn_train,Tn_train);
% 
% %% 仿真测试
% Tn_sim = sim(net,Pn_test);    
% % 反归一化
% T_sim = mapminmax('reverse',Tn_sim,outputps);
% 
% %% 结果对比
% result = [T_test' T_sim'];
% % 均方误差
% E = mse(T_sim - T_test);
% 
% error=T_sim-T_test;
% error_optimized=T_sim_optimized - T_test;
% 
% comparedPredictedValue=[error_optimized' error'];
genFunction(net_optimized,'bsGAModel');
% genFunction(net_optimized,'bsLowerGAModel');