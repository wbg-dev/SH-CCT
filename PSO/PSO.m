%% 清空环境
clear all
clc
global trainFcn
trainFcn='trainbr';
load("lowerData.mat");
load bsValForBook;
%训练数据和预测数据
input_train=P_train.';
output_train=T_train.';
input_test=P_lowerVal.';
output_test=T_val.';
%节点个数
inputnum=size(input_train,1);
hiddennum=10;
outputnum=1;
length=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%构建网络
netPSO=newff(inputn,outputn,hiddennum);

% 参数初始化
%粒子群算法中的两个参数
c1 = 1.49445;
c2 = 1.49445;

maxgen=100;   % 进化次数  
sizepop=40;   %种群规模

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-popmax;

for i=1:sizepop
    pop(i,:)=popmax*rands(1,length);
    V(i,:)=rands(1,length);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,netPSO,inputn,outputn);
end
% 个体极值和群体极值
[bestfitness, bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest=pop;    %个体最佳
fitnessgbest=fitness;   %个体最佳适应度值
fitnesszbest=bestfitness;   %全局最佳适应度值

%% 迭代寻优
for i=1:maxgen
    for j=1:sizepop
        %速度更新
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        %种群更新
        pop(j,:)=pop(j,:)+V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        %自适应变异
        pos=unidrnd(length);
        if rand>0.95
            pop(j,pos)=popmax*rands(1,1);
        end
        %适应度值
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,netPSO,inputn,outputn);
    end
    
    for j=1:sizepop
    %个体最优更新
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %群体最优更新 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    end
    yy(i)=fitnesszbest;    
        
end
x=zbest;
%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

netPSO.iw{1,1}=reshape(w1,hiddennum,inputnum);
netPSO.lw{2,1}=reshape(w2,outputnum,hiddennum);
netPSO.b{1}=reshape(B1,hiddennum,1);
netPSO.b{2}=B2;

%% BP网络训练
%网络进化参数
netPSO.trainParam.epochs=10000;
netPSO.trainParam.lr=0.01;
netPSO.trainParam.goal=1e-7;
netPSO.trainFcn=trainFcn;

%网络训练
[netPSO,per2]=train(netPSO,inputn,outputn);

%% BP网络预测
%数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(netPSO,inputn_test);
test_PSO=mapminmax('reverse',an,outputps);
errorPSO=test_PSO-output_test;
net = newff(inputn,outputn,hiddennum);

% 设置训练参数
net.trainParam.epochs = 10000;
net.trainParam.show = 10;
net.trainParam.goal = 1e-7;
net.trainParam.lr = 0.1;
% net.layers{1}.transferFcn=transferFcn1;
% net.layers{2}.transferFcn=transferFcn2;
net.trainFcn=trainFcn;
[net,per3] = train(net,inputn,outputn);

%% 仿真测试   
% 反归一化
an1=sim(net,inputn_test);
T_sim = mapminmax('reverse',an1,outputps);
result = [output_test' T_sim'];
% 均方误差

error=T_sim-output_test;
PSOMSE=mse(errorPSO);
comparedPredictedValue=[errorPSO' error'];
% genFunction(net,'bsPSOModelLM');
% if strcmp(trainFcn,'trainlm')
%     genFunction(net,'bsPSOModelLM');
% end
% if strcmp(trainFcn,'trainbr')
%     genFunction(net,'bsPSOModelBR');
% end
% view(net);
