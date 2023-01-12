function[W1,B1,W2,B2,val]=gadecod(x)
global P_train
global T_train
global inputNum
global outputNum
global hiddenLayerSize;
% 前输入*隐含个编码为W1
for i=1:hiddenLayerSize
    for k=1:inputNum
        W1(i,k)=x(inputNum*(i-1)+k);
    end
end
% 接着的S1*S2个编码(即第R*S1个后的编码)为W2
for i=1:outputNum
    for k=1:hiddenLayerSize
        W2(i,k)=x(hiddenLayerSize*(i-1)+k+inputNum*hiddenLayerSize);
    end
end
% 接着的S1个编码(即第R*SI+SI*S2个后的编码)为B1
for i=1:hiddenLayerSize
    B1(i,1)=x((inputNum*hiddenLayerSize+hiddenLayerSize*outputNum)+i);
end
%接着的S2个编码(即第R*SI+SI*S2+S1个后的编码)为B2
for i=1:outputNum
    B2(i,1)=x((inputNum*hiddenLayerSize+hiddenLayerSize*outputNum+hiddenLayerSize)+i);
end
% 计算S1与S2层的输出
A1=logsig(W1*P_train,B1);
A2=purelin(W2*A1,B2);
% 计算误差平方和
SE=sumsqr(T_train-A2);
% 遗传算法的适应值
val=1/SE;