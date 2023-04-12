% deepL_multiple_inputs_v1
% 
% based on deepL_calibration_v1.m and
% TrainNetworkOnImageAndFeatureDataExample.mlx
% 
% 
%
% Michael Alvarez
% michael.alvarez2@upr.edu

close all
clear
clc

%% Load data
% load BIDMC dataset in MAT array
load('bidmc_data.mat') % data: output 1x53 struct 

% preview data
for k = 1 %: size(data,2)
    
ppg = {data(k).ppg}.'; % take single ppg signal from data array
plot(ppg{1,1}.v)       % plot ppg signal

spo2v ={data(k).ref.params.spo2.v}' ; % take the oxygen saturation vector
spo2 = nanmean(spo2v{1,1}) ;          % compute the mean without nan values

title("PPG id: " + k + " - SpO2: " + spo2) % add informative title

drawnow 
pause(.5)
end
    
%% Create a subset
disp('Create a subset')
fs = 125 ; % sample rate

T = 2 ; % seconds of the signal to take

cont = 0 ;

max_samples = 25440 ; % know because run the script withou save data
inputs  = zeros( 250 , max_samples ) ;
targets = zeros( 1 , max_samples ) ;

samplesDataset = size(data,2) ;
for k = 1 : samplesDataset
    disp( " iter: " + k + "/" + samplesDataset )
    
    ppg   = {data(k).ppg}.' ;
    spo2v = {data(k).ref.params.spo2.v}' ;

    length_ppg  = size(ppg{1,1}.v,1) ; % in case of variable size
    length_spo2 = size(spo2v{1,1},1) ;

    num_inter  = floor( length_ppg / ( fs * T ) ) ; 
    
    slope_spo2_inter = ( length_spo2 - 1 ) / ( num_inter ) ;
    
    for ki = 1 :num_inter
        
        index = ( 1 : fs * T ) + fs * T * ( ki - 1 ) ;
        ppg_cut = ppg{1,1}.v( index ) ;
        
        indexSo = floor( slope_spo2_inter * ( ki - 1 ) + 1 ) ;
        indexSf = floor( slope_spo2_inter * ( ki     ) + 1 ) ;
        
        index_spo2 = indexSo:indexSf ;        
        spo2_cut = nanmean( spo2v{1,1}( index_spo2 ) ) ;

        % check if there is nan in SpO2:
        if isnan( spo2_cut )
            break
        else
            % input , target
            cont = cont + 1 ;
            inputs(:,cont)  = ppg_cut ;
            targets(1,cont) = spo2_cut ;
        end
        
    end
% plot(ppg{1,1}.v)

% spo2 = nanmean(spo2v{1,1}) ;

% title(["PPG id: " + k + " - SpO2: " + spo2])
% drawnow
% pause(.5)
end
disp(["Dataset has " + cont + "samples"])

% remove empty rows:
dataInput  = inputs( : , 1 : cont ) ;
dataTarget = targets( : , 1 : cont ) ;

clear inputs targets

%% Training and Testing dataset
disp('Training and Testing dataset')
numsamples =  cont ; 
thIndex = floor( .8 * numsamples ) ;

rng(1)
randomIndex = randperm( cont ) ;

clear XTrain YTrain    
XTrain = cell(thIndex,1) ;
YTrain = zeros(thIndex,1) ;
for kf = 1 :thIndex   
    XTrain{kf,1} = dataInput(:,randomIndex(kf)) ; 
    YTrain(kf,1) = dataTarget(1,randomIndex(kf)) ; 
end
    
clear XTest YTest
XTest = cell( numsamples - thIndex , 1 ) ;
YTest = zeros( numsamples - thIndex , 1 ) ;
for kf =  thIndex + 1 : numsamples  
    XTest{kf-thIndex,1} = dataInput( : , randomIndex( kf ) ) ; 
    YTest(kf-thIndex,1)  = dataTarget( 1 , randomIndex( kf ) ) ; 
end
    
%% Deep Learning model
disp('Deep Learning model')

numFeatures = size( XTrain{1,1} , 1 ) ;
numHiddenUnits = 150;
numResponses = 1;    

% DL model
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% options
maxEpochs = 200;
miniBatchSize = 10;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the model with the specified training options.
net = trainNetwork(XTrain,YTrain,layers,options);

%% Testing
disp('Testing')

outN = predict(net,XTest) ;

difference = outN - YTest ;
meanError = mean( abs( difference ) ) ;
variError = var( abs( difference ) ) ;

plot(difference,'.r')

disp('Test Deep learning model: ')
disp('Target:')
disp(YTest')
disp('Deep model:')
disp(outN')
disp('Difference:')
disp(difference')
disp(["Error: "+ meanError + " +- " + variError + " % "])
