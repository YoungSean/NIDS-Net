# NIDS-Net
The approach for Novel Instance Detection and Segmentation (NIDS)

## Getting Started

### Prerequisites
- Python 3.7 or higher (tested 3.9)
- torch (tested 2.0)
- torchvision

### Installation
```sh
git clone https://github.com/YoungSean/NIDS-Net.git
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python setup.py install
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
#### Download [ViT-H SAM weights](https://github.com/facebookresearch/segment-anything#model-checkpoints)
```shell
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
After installation, there will be a folder named "ckpts". Move the SAM weight to "ckpts/sam_weights/sam_vit_h_4b8939.pth".


### Preparing Datasets
We do not need training datasets for detectors. We can use template embeddings to train the adapter.

#### High-resolution Dataset
This [instance-detection repo](https://github.com/insdet/instance-detection) provide the [InsDet-Full](https://drive.google.com/drive/folders/1rIRTtqKJGCTifcqJFSVvFshRb-sB0OzP).
```shell
cd $ROOT
ln -s $HighResolution_DATA database
```
We provide the preprocessed testing images in this [link](https://utdallas.box.com/s/bfcgn0dpbvu5w5be20wyj41fzsjajh4h) accorinding to this [instance-detection](https://github.com/insdet/instance-detection). Please put them into "Data" folder as follows:
```
database
│
└───Background
│
└───Objects
│   │
│   └───000_aveda_shampoo
│   │   │   images
│   │   │   masks
│   │
│   └───001_binder_clips_median
│       │   images
│       │   masks
│       │   ...
│   
│   
└───Data
    │   test_1_all
    │   test_1_easy
    │   test_1_hard
```

#### RoboTools, LM-O and YCB-V
VoxDet provides the [datasets](https://github.com/Jaraxxus-Me/VoxDet).
Save and unzip them in '$ROOT/datasets' to get "datasets/RoboTools", "datasets/lmo", "datasets/ycbv". 

## Usage
1. Check GroundingDINO and SAM
- SAM: [`test_sam.py`](test_sam.py)
- GroundingDINO + SAM: [`test_gdino_sam.py`](test_gdino_sam.py)

2. Generate template embeddings via get_object_features_via_FFA.py.
   Or you can download the [template embeddings](https://utdallas.box.com/s/ieo7lochg1dzzdjfqm7saiudaeptufoi). You may adjust their filename to load them in the python scripts.
3. Train weight adapters in adapter.py (Optional).
You can try the basic version without the weight adapter.
4. Inference 
```sh
# for high-resolution dataset
# demo image
python demo_eval_gdino_FFA.py

# dataset results
python mini_test_eval_gdino_FFA.py
# for lmo 
python lmo_test_eval_gdino_FFA.py
# since YCB-V and RoboTools have scenes
# we first get detection prediction results for each scene
./get_ycbv_prediction.sh 
./get_RoboTools_prediction.sh 

# then merge them using utils/merge_COCO_json.py
# evaluate them with eval_result.py

```

## BOP segmentation code will be released soon.

## Acknowledgments

This project is based on the following repositories:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [CLIP](https://github.com/openai/CLIP)
- [VoxDet](https://github.com/Jaraxxus-Me/VoxDet)
- [InsDet](https://github.com/insdet/instance-detection)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [DINOv2](https://github.com/facebookresearch/dinov2)


