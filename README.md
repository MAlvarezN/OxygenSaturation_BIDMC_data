# OxygenSaturation_BIDMC_data
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

## DL Model
In this instructive reposotory we present a Deep Learning model (based on [PP-net](https://doi.org/10.1109/JSEN.2020.2990864)) to estimate the Oxygen saturation from a single PPG signal.

![Example DL model](https://github.com/MAlvarezN/OxygenSaturation_BIDMC_data/blob/14c87e105f57e630471ece22c3208e59ad23e67a/ModelBasedPPNet_h.PNG)

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
The mean error for testing samples is 16.06, with a few extreme values with error close to 28%.

