function[W1,B1,W2,B2,val]=gadecod(x)
global P_train
global T_train
global inputNum
global outputNum
global hiddenLayerSize;
% ǰ����*����������ΪW1
for i=1:hiddenLayerSize
    for k=1:inputNum
        W1(i,k)=x(inputNum*(i-1)+k);
    end
end
% ���ŵ�S1*S2������(����R*S1����ı���)ΪW2
for i=1:outputNum
    for k=1:hiddenLayerSize
        W2(i,k)=x(hiddenLayerSize*(i-1)+k+inputNum*hiddenLayerSize);
    end
end
% ���ŵ�S1������(����R*SI+SI*S2����ı���)ΪB1
for i=1:hiddenLayerSize
    B1(i,1)=x((inputNum*hiddenLayerSize+hiddenLayerSize*outputNum)+i);
end
%���ŵ�S2������(����R*SI+SI*S2+S1����ı���)ΪB2
for i=1:outputNum
    B2(i,1)=x((inputNum*hiddenLayerSize+hiddenLayerSize*outputNum+hiddenLayerSize)+i);
end
% ����S1��S2������
A1=logsig(W1*P_train,B1);
A2=purelin(W2*A1,B2);
% �������ƽ����
SE=sumsqr(T_train-A2);
% �Ŵ��㷨����Ӧֵ
val=1/SE;