%% ��ջ���
clear all
clc
global trainFcn
trainFcn='trainbr';
load("lowerData.mat");
load bsValForBook;
%ѵ�����ݺ�Ԥ������
input_train=P_train.';
output_train=T_train.';
input_test=P_lowerVal.';
output_test=T_val.';
%�ڵ����
inputnum=size(input_train,1);
hiddennum=10;
outputnum=1;
length=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%��������
netPSO=newff(inputn,outputn,hiddennum);

% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;
c2 = 1.49445;

maxgen=100;   % ��������  
sizepop=40;   %��Ⱥ��ģ

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-popmax;

for i=1:sizepop
    pop(i,:)=popmax*rands(1,length);
    V(i,:)=rands(1,length);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,netPSO,inputn,outputn);
end
% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness, bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest=pop;    %�������
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for i=1:maxgen
    for j=1:sizepop
        %�ٶȸ���
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        %��Ⱥ����
        pop(j,:)=pop(j,:)+V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        %����Ӧ����
        pos=unidrnd(length);
        if rand>0.95
            pop(j,pos)=popmax*rands(1,1);
        end
        %��Ӧ��ֵ
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,netPSO,inputn,outputn);
    end
    
    for j=1:sizepop
    %�������Ÿ���
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %Ⱥ�����Ÿ��� 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    end
    yy(i)=fitnesszbest;    
        
end
x=zbest;
%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

netPSO.iw{1,1}=reshape(w1,hiddennum,inputnum);
netPSO.lw{2,1}=reshape(w2,outputnum,hiddennum);
netPSO.b{1}=reshape(B1,hiddennum,1);
netPSO.b{2}=B2;

%% BP����ѵ��
%�����������
netPSO.trainParam.epochs=10000;
netPSO.trainParam.lr=0.01;
netPSO.trainParam.goal=1e-7;
netPSO.trainFcn=trainFcn;

%����ѵ��
[netPSO,per2]=train(netPSO,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(netPSO,inputn_test);
test_PSO=mapminmax('reverse',an,outputps);
errorPSO=test_PSO-output_test;
net = newff(inputn,outputn,hiddennum);

% ����ѵ������
net.trainParam.epochs = 10000;
net.trainParam.show = 10;
net.trainParam.goal = 1e-7;
net.trainParam.lr = 0.1;
% net.layers{1}.transferFcn=transferFcn1;
% net.layers{2}.transferFcn=transferFcn2;
net.trainFcn=trainFcn;
[net,per3] = train(net,inputn,outputn);

%% �������   
% ����һ��
an1=sim(net,inputn_test);
T_sim = mapminmax('reverse',an1,outputps);
result = [output_test' T_sim'];
% �������

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
