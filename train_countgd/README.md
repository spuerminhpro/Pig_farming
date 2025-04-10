# [NeurIPS 2024] CountGD: Multi-Modal Open-World Counting

Niki Amini-Naieni, Tengda Han, & Andrew Zisserman

Official PyTorch implementation for CountGD. Details can be found in the paper, [[Paper]](https://arxiv.org/abs/2407.04619) [[Project page]](https://www.robots.ox.ac.uk/~vgg/research/countgd/).

For Inference code:

[CountGD Repository](https://github.com/spuerminhpro/Pig_farming/tree/main/countgd)

## Contents
* [Preparation](#preparation)
* [CountGD Train](#countgd-train)
* [CountBench](#countbench)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Preparation
### 1. Prepare Dataset
[Dataset Documentation](Dataset_doc.md)
open notebook to create train and test dataset
```
create_dataset_countgd.ipynb
```

### 2. Install GCC

Install GCC. In this project, GCC 11.3 and 11.4 were tested. The following command installs GCC and other development libraries and tools required for compiling software in Ubuntu.

```
sudo apt update
sudo apt install build-essential
```

### 3. Clone Repository

```
git clone https://github.com/spuerminhpro/Pig_farming/tree/main/train_countgd
```

### 4. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running the CountGD training and inference procedures. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n countgd python=3.11
conda activate countgd
cd CountGD
pip install -r requirements.txt
cd models/GroundingDINO/ops
python setup.py build
pip install .
python test.py # should result in 6 lines of * True
pip install git+https://github.com/facebookresearch/segment-anything.git
cd ../../../
```

### 5. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```CountGD``` repository.

  ```
  mkdir checkpoints
  ```

* Execute the following command.

  ```
  python download_bert.py
  ```

* Download the pretrained Swin-B GroundingDINO weights.

  ```
  wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
  ```

* Download the pretrained ViT-H Segment Anything Model (SAM) weights.

  ```
  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```
* Download the pretrained countgd weight from paper.
  [Google Drive link (1.2 GB)](https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI/view?usp=sharing)

## CountGD Train
setput dataset path in 'custome_data/custome_dataset.json'

[Custom Dataset](custome_data/custome_dataset.json)

Modify model parameter in config/cfg_fsc147_vit_b_odvg.py
```
  epoch= 
  batch_size=
  label_list = ['pig']  
  val_label_list = ['pig']
  ```
Train
```
python main.py --output_dir ./gdino_train -c config/cfg_fsc147_vit_b_odvg.py --datasets custome_data/custome_dataset.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased 
```
Test
```
python main.py --output_dir ./gdino_test -c config/cfg_fsc147_vit_b_test.py --eval --datasets config/datasets_fsc147.json --pretrain_model_path ./logs_8_2_2024_thresh_0.23_lr_1e-4_vit_b/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased
```

