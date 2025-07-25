# Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting

This is the expanding work from the original paper "Rethinking Fourier Transform from A Basis Functions Perspective for Long-term Time Series Forecasting." (NeurIPS 2024) to a journal. 
### This is the offical implementation of FBM-S model. 

### Implement the project

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download the ETTh1, ETTh2, ETTm1, ETTm2, Electricity and Traffic data from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and WTH data from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) The other datasets can be download at [Baidu Drive](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/long_term_forecast/file_to_implement.sh``` and ```./scripts/short_term_forecast/file_to_implement.sh```
```
sh ./scripts/long_term_forecast/ETTh1.sh
sh ./scripts/short_term_forecast/PEMS.sh
```
You can adjust the hyperparameters based on your needs.
## Fourier Basis Mapping

![alt text](https://github.com/runze1223/FBM-S/blob/main/pic/imag1.png)
![alt text](https://github.com/runze1223/FBM-S/blob/main/pic/imag2.png)

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/yuqinie98/PatchTST

https://github.com/ServiceNow/N-BEATS

https://github.com/aikunyi/FreTS

https://github.com/hqh0728/CrossGNN

https://github.com/thuml/iTransformer

https://github.com/kwuking/TimeMixer

https://github.com/VEWOXIC/FITS

https://github.com/decisionintelligence/DUET

## Citation

If you find this repository useful, please consider citing our paper. Give a star to support this respiratory.

If you have any questions, feel free to contact: runze.y@sjtu.edu.cn
