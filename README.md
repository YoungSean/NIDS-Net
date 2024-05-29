# NIDS-Net
A unified framework for Novel Instance Detection and Segmentation (NIDS).

[arXiv](https://arxiv.org/abs/2405.17859)

## Adapting Pre-Trained Vision Models for Novel Instance Detection and Segmentation
> Novel Instance Detection and Segmentation (NIDS) aims at detecting and segmenting novel object instances given a few examples of each instance. We propose a unified framework (NIDS-Net) comprising object proposal generation, embedding creation for both instance templates and proposal regions, and embedding matching for instance label assignment. Leveraging recent advancements in large vision methods, we utilize the Grounding DINO and Segment Anything Model (SAM) to obtain object proposals with accurate bounding boxes and masks. Central to our approach is the generation of high-quality instance embeddings. We utilize foreground feature averages of patch embeddings from the DINOv2 ViT backbone, followed by refinement through a weight adapter mechanism that we introduce. We show experimentally that our weight adapter can adjust the embeddings locally within their feature space and effectively limit overfitting. This methodology enables a straightforward matching strategy, resulting in significant performance gains. Our framework surpasses current state-of-the-art methods, demonstrating notable improvements of 22.3, 46.2, 10.3, and 24.0 in average precision (AP) across four detection datasets. In instance segmentation tasks on seven core datasets of the BOP challenge, our method outperforms the top RGB methods by 3.6 AP and remains competitive with the best RGB-D method.
## Framework

![NIDS-Net.](imgs/fw0.png)

## Example
![Demo detection results on real datasets, High-resolution and RoboTools.](imgs/det6.png)


## Getting Started
We prepare demo google colabs: [inference on a high-resolution image](https://colab.research.google.com/drive/1dtlucQ5QryLgooSDkH-Qumxrrnb-9FCg?usp=sharing) and [Training free one-shot detection](https://colab.research.google.com/drive/1IM8TgpNo_9TijopO3PRyZea7MgTUjv30?usp=sharing). 
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
```shell
mkdir ckpts/sam_weights
mv sam_vit_h_4b8939.pth ckpts/sam_weights
```

### Preparing Datasets
<details>
<summary> Setting Up 4 Detection Datasets </summary>
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
</details>


## Usage
You can directly use the demo google colabs: [inference on a high-resolution image](https://colab.research.google.com/drive/1dtlucQ5QryLgooSDkH-Qumxrrnb-9FCg?usp=sharing) and [Training free one-shot detection](https://colab.research.google.com/drive/1Wam974xV82oq-uLbnWstsDSaoEpSWvP1?usp=sharing).

1. Check GroundingDINO and SAM
- SAM: [`test_sam.py`](test_sam.py)
- GroundingDINO + SAM: [`test_gdino_sam.py`](test_gdino_sam.py)

2. Generate template embeddings via get_object_features_via_FFA.py.
Or you can download the [template embeddings](https://utdallas.box.com/s/ieo7lochg1dzzdjfqm7saiudaeptufoi). You may adjust their filenames to load them in the python scripts.
```shell
mkdir obj_FFA
wget https://utdallas.box.com/shared/static/50a8q7i5hc33rovgyavoiw0utuduno39 -O obj_FFA/object_features_vitl14_reg.json

mkdir BOP_obj_feat
wget https://utdallas.box.com/shared/static/qlyekivfg6svx84xhh5xv39tun3xza1u -O BOP_obj_feat/lmo_object_features.json
wget https://utdallas.box.com/shared/static/keilpt2i2gk0rrjymg0nkf88bdr734wm -O BOP_obj_feat/ycbv_object_features.json
mkdir RoboTools_obj_feat
wget https://utdallas.box.com/shared/static/e7o7fy00qitmbyg51wb6at9vc1igzupb -O RoboTools_obj_feat/object_features.json
mkdir adapted_obj_feats
```
3. Train weight adapters in adapter.py (Optional).
You can try the basic version without the weight adapter.

To train the adapter, you need to prepare the training dataset and set some parameters in the script. The following is an example of training the weight adapter for the high-resolution dataset.

After training the adapter, you can use the adapter to fine-tune the template embeddings to save them into the folder like './adapted_obj_feats'. The script will save the adapted template embeddings in the same format as the original template embeddings. 

With the trained adapter, You can also use utils/transform_adapted_feats.py to transform the original template embeddings to the adapted template embeddings. It was used for the common adapter of BOP datasets.

To reuse the adapter, you can set the flag "use_adapter" to True and set the adapter weight path in the inference scripts. 
```shell
python adapter.py
```
4. Inference 
```sh
# for high-resolution dataset
# demo image
# in each script, there are some parameters you can adjust
# for example, the flag "use_adapter", the adapter type and the adapter weight path in demo_eval_gdino_FFA.py

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

## BOP segmentation code will be released soon. (TO DO)

## Citation
If you find the method useful in your research, please consider citing:
```latex
@misc{lu2024adapting,
      title={Adapting Pre-Trained Vision Models for Novel Instance Detection and Segmentation}, 
      author={Yangxiao Lu and Jishnu Jaykumar P and Yunhui Guo and Nicholas Ruozzi and Yu Xiang},
      year={2024},
      eprint={2405.17859},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

This project is based on the following repositories:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [CLIP](https://github.com/openai/CLIP)
- [VoxDet](https://github.com/Jaraxxus-Me/VoxDet)
- [InsDet](https://github.com/insdet/instance-detection)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [SAM6D](https://github.com/JiehongLin/SAM-6D)


