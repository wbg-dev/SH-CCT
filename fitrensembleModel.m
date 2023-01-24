clear all
clc
% load ydData2
% % load elementName
load data
X = BsTrainInput;
Y = BsTrainOutput;
X_test=BsTestInput;
Y_test=BsTestOutput;
rng default
% Optimizer :
% 搜索算法
% bayesopt — Use Bayesian optimization.
% gridsearch — Use grid search with NumGridDivisions values per dimension.
% randomsearch — Search at random among MaxObjectiveEvaluations points.
% AcquisitionFunctionName:
% 采集函数
% EI
%'expected-improvement-per-second-plus'
%'expected-improvement'
%'expected-improvement-plus'
%'expected-improvement-per-second'
%'lower-confidence-bound'
% PI
%'probability-of-improvement'
% MaxObjectiveEvaluations
% 贝叶斯和随机是默认30，网格全部
% NumGridDivisions
% 网格数
% Repartition
% 去噪声用true
% UseParallel
% 采用并行池 用true ,'kfold',10
Mdl = fitrensemble(X,Y,...
    'Method','Bag',...
    'Learner',templateTree('Surrogate','on'),...
    'OptimizeHyperparameters',{'Method','NumLearningCycles','LearnRate','MinLeafSize','MaxNumSplits'},...
    'HyperparameterOptimizationOptions',struct('Optimizer','bayesopt','MaxObjectiveEvaluations',30,'Repartition',true,...
    'AcquisitionFunctionName','expected-improvement','kfold',10));

% predictedValueTrain=predict(Mdl,input);
% errorTrain=predictedValueTrain-output;
% mseTrainPredicted=mse(errorTrain);
% maeTrainPredicted=mae(errorTrain);

predictedValueTrain=predict(Mdl,X);
errorTrain=predictedValueTrain-Y;
mseTrainPredicted=mse(errorTrain);
maeTrainPredicted=mae(errorTrain);

predictedValueTest=predict(Mdl,X_test);
errorTest=predictedValueTest-Y_test;
mseTestPredicted=mse(errorTest);
maeTestPredicted=mae(errorTest);

% predictedValueVal=predict(Mdl,P_val);
% errorVal=predictedValueVal-T_val;
% mseValPredicted=mse(errorVal);
% maeValPredicted=mae(errorVal);

% [R2ofTest,RofTest] = rSquareAndR(predictedValueTest,T_test);
% [R2ofTrain,RofTrain] = rSquareAndR(predictedValueTrain,output);
% [R2ofVal,RofVal] = rSquareAndR(predictedValueVal,T_val);
% plotPartialDependence(Mdl,1)
% [imp,ma] = predictorImportance(Mdl);
% elementImp=imp(1,1:17);
% totalElement=sum(elementImp);
% stdElementImp=elementImp/totalElement;
% pie(stdElementImp,elementName)
% legend(elementName);
% sound(sin(2*pi*25*(1:4000)/100));
%plotPartialDependence(Mdl,1)
