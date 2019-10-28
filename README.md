##  Deep Constrained Dominant Sets for Person Re-Identification [(DCDS)](https://www.crcv.ucf.edu/wp-content/uploads/2019/08/Publications_Deep-Constrained-Dominant-Sets-for-Person-Re-Identification.pdf)
Pytorch Implementation for our **ICCV2019** work, Deep Constrained Dominant Sets for Person Re-Identification.
This implementation is based on [open-reid](https://github.com/Cysu/open-reid) and [kpm_rw_person_reid.](https://github.com/YantaoShen/kpm_rw_person_reid)

### Requirements 
* [python 2.7](https://www.python.org/download/releases/2.7/) 
* [PyTorch](https://pytorch.org/previous-versions/) (we run the code under version 0.3.0)
* [metric-learn 0.3.0](https://pypi.org/project/metric-learn/0.3.0/)  
* [torchvision 0.2.1](https://pypi.org/project/torchvision/0.2.1/)

```shell 
- cd DCDS
- install setup.py
```


For single dataset (SD) setup we use [Market1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [CUHK03](//docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0), and [DukeMTMC](https://drive.google.com/uc?id=0B0VOCNYh8HeRdnBPa2ZWaVBYSVk) datasets.
### Download and extract these datasets and do,
```shell 
cd examples/
mkdir data
cd data/
mkdir market1501
cd market1501
mkdir raw/
mv dir_of_market1501_zip raw/
```
Repeate this for CUHK03 and Dukemtmc.



## Example

### Training
```
python examples/main.py -d market1501 -b 64 -a resnet101 --features 2048 --lr 0.0001 --ss 10 --epochs 100 --dropout 0 --weight-decay 0 --logs-dir examples/logs/market1501-final-model
```
```
python examples/main.py -d dukemtmc -b 64 -a resnet101 --features 2048 --lr 0.0001 --ss 10 --epochs 100 --dropout 0 --weight-decay 0 --logs-dir examples/logs/dukemtmc-final-model
```

- Same for CUHK03

### Testing
Download the trained models
* [Market1501](https://drive.google.com/file/d/14wYOpiPD7O1ETyqY9d5ABUpWZTNebORB/view?usp=sharing)
* [DukeMTMC](https://drive.google.com/file/d/1NJV7DOqiwan51W0aPl2OlPwKQPcZVuwB/view?usp=sharing)
* [CUHK03](https://drive.google.com/file/d/1sqm2Lw18hRP2YH_lZmzM9xLUDfIxbCnM/view?usp=sharing)
```
python examples/main.py -d market1501 -b 64 -a resnet101 --features 2048 --evaluate --evaluate-from examples/logs/market1501-final-model/model_best.pth.tar

python examples/main.py -d dukemtmc -b 64 -a resnet101 --features 2048 --evaluate --evaluate-from examples/logs/dukemtmc-final-model/model_best.pth.pth.tar
```
## Results, trained and tested on single dataset, SD setup.
<table>
  <tr>
    <th></th>
    <th colspan="3">Market-1501</th>
    <th colspan="3">CUHK03</th>
    <th colspan="3">DukeMTMC-reID</th>
  </tr>
  <tr>
    <th></th>
    <th>mAP(%)</th><th>rank-1</th><th>rank-5</th>
    <th>mAP(%)</th><th>rank-1</th><th>rank-5</th>
    <th>mAP(%)</th><th>rank-1</th><th>rank-5</th>
  </tr>
  <tr>
    <th>DCDS (SD)</th>
    <th>81.5</th><th>92.9</th><th>97.4</th>
    <th>90.7</th><th>93.3</th><th>99.1</th>
    <th>70.3</th><th>83.6</th><th>90.4</th>
  </tr>
 
</table>

## Citation
```
@InProceedings{Alemu_2019_ICCV,
author = {Alemu, Leulseged Tesfaye and Pelillo, Marcello and Shah, Mubarak},
title = {Deep Constrained Dominant Sets for Person Re-Identification},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
