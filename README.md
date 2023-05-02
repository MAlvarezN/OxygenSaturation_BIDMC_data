# OxygenSaturation_BIDMC_data Predictio Using DNN
This repository contains instructions to estimate Oxygen Saturation from BIDMC data using simple Deep Learning models


## Data
### BIDMC PPG and Respiration Dataset
This dataset contains signals and numerics extracted from the much larger MIMIC II matched waveform Database, along with manual breath annotations made from two annotators, using the impedance respiratory signal.
https://physionet.org/content/bidmc/1.0.0/

### How preview data
```matlab
% load BIDMC dataset in MAT array
load('bidmc_data.mat') % data: output 1x53 struct 

% preview data
for k = 1 : size(data,2)
    
ppg = {data(k).ppg}.'; % take single ppg signal from data array
plot(ppg{1,1}.v)       % plot ppg signal

spo2v ={data(k).ref.params.spo2.v}' ; % take the oxygen saturation vector
spo2 = nanmean(spo2v{1,1}) ;          % compute the mean without nan values

title("PPG id: " + k + " - SpO2: " + spo2) % add informative title

drawnow 
pause(.5)
end
```

#### Example:
Figure shows a PPG signals took from the first register of BIDMC dataset. The cut-interval is randomly selected as example with three pulses. The x-axis represents the samping points and y-axis the intensity registered.
![PPG signal from BIDMC dataset](https://github.com/MAlvarezN/OxygenSaturation_BIDMC_data/blob/553684e8563484b63c21b6b571f4b1715a832f29/sample_PPGsignal.png)

## DL Models

In this instructive reposotory we present Deep Learning models: (a) Model based on [PP-net](https://doi.org/10.1109/JSEN.2020.2990864), and (b) LSTM model; to estimate the Oxygen saturation from a single PPG signal.

(a) Model based on [PP-net](https://doi.org/10.1109/JSEN.2020.2990864)
![Example DL model](https://github.com/MAlvarezN/OxygenSaturation_BIDMC_data/blob/14c87e105f57e630471ece22c3208e59ad23e67a/ModelBasedPPNet_h.PNG)

(b) LSTM model

<img src="https://github.com/MAlvarezN/OxygenSaturation_BIDMC_data/blob/0f2baba542a47eabb1ed88e17e4b58b08bfd229a/Model_lstm_h.PNG" width="350" height="140">

### Parameters

```matlab
maxEpochs = 200;
miniBatchSize = 10;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
```
### Results
(a) Model based on [PP-net](https://doi.org/10.1109/JSEN.2020.2990864)

The mean error for testing samples is 16.06%, with few outlier values with error close to 28%.

<img src="https://github.com/MAlvarezN/OxygenSaturation_BIDMC_data/blob/b1261d38ed213d3a5d44924be3bab641ab95a2ad/Error_based_PPnet.png" width="600" height="500">

(b) LSTM model
The mean error for testing samples is 2.39%, with few outlier values with error close to 15%.

<img src="https://github.com/MAlvarezN/OxygenSaturation_BIDMC_data/blob/817c17792945933ffa907291f03b1e2698ba8427/Error_LSTM.png" width="600" height="500">
