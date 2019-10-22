%-------------------------------------------------------------------------%
%  Binary Grey Wolf Optimization (BGWO) source codes demo version         %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

clc, clear, close 
% Benchmark data set contains 351 instances and 34 features (binary class)
load ionosphere.mat; % Matlab also provides this dataset (load Ionosphere.mat)
% Call features & labels
feat=f; label=l; 
%---Input------------------------------------------------------------------
% feat:  feature vector (instances x features)
% label: labelling
% N:     Number of wolves
% T:     Maximum number of iterations
% *Note: k-value of KNN & k-fold setting can be modified in jFitnessFunction.m
%---Output-----------------------------------------------------------------
% sFeat: Selected features (instances x features)
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------



%% (Method 2) BGWO2
close all; N=10; T=100; 
[sFeat,Sf,Nf,curve]=jBGWO2(feat,label,N,T); 

% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of Iterations');
ylabel('Fitness Value'); title('BGWO2'); grid on;





