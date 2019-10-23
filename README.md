#  Deep Constrained Dominant Sets for Person Re-Identification (DCDS)
## Pytorch Implementation for our ICCV2019 work called Deep Constrained Dominant Sets for Person Re-Identification
This implementation is based on Oper Reid and kpm_rw_person_reid
### Requirements 
* python 2.7 
* PyTorch (we run the code under version 0.3.0)
* metric-learn 0.3.0
* torchvision 0.2.1


- Cd DCDS
- install setup.py


For single dataset (SD) setup we use Market1501, CUHK03, and DukeMTMC datasets.
### Download and extract these dataset,

cd examples/
mkdir data
cd data/
mkdir market1501
cd market1501
mkdir raw/
mv dir_of_market1501_zip raw/

Repeate this for Market1501 and Dukemtmc.



## Example

### Training

-d market1501 -b 64 -a resnet101 --features 2048 --lr 0.0001 --ss 10 --epochs 100 --dropout 0 --weight-decay 0 --logs-dir examples/logs/market1501-final-model

--d dukemtmc -b 64 -a resnet101 --features 2048 --lr 0.0001 --ss 10 --epochs 100 --dropout 0 --weight-decay 0 --logs-dir examples/logs/dukemtmc-final-model


- Same for CUHK03

### Testing

-d market1501 -b 64 -a resnet101 --features 2048 --evaluate --evaluate-from examples/logs/market1501-final-model/checkpoint.pth.tar





