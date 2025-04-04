# [NeurIPS 2024] CountGD: Multi-Modal Open-World Counting

Niki Amini-Naieni, Tengda Han, & Andrew Zisserman

Official PyTorch implementation for CountGD. Details can be found in the paper, [[Paper]](https://arxiv.org/abs/2407.04619) [[Project page]](https://www.robots.ox.ac.uk/~vgg/research/countgd/).

If you find this repository useful, please give it a star ⭐.

## Try Using CountGD to Count with Text, Visual Exemplars, or Both Together Through the App [[HERE]](https://huggingface.co/spaces/nikigoli/countgd).

## Try Out the Colab Notebook to Count Objects in All the Images in a Zip Folder With Text [[HERE]](https://huggingface.co/spaces/nikigoli/countgd/blob/main/notebooks/demo.ipynb).

## Contents
* [Preparation](#preparation)
* [CountGD Train](#countgd-train)
* [CountBench](#countbench)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Preparation
### 1. Download Dataset

In our project, the FSC-147 dataset is used.
Please visit following link to download this dataset.

* [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

Create custome dataset:

  Convert data from Yolo format in create_dataset_countgd.ipynb
  
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
conda create -n countgd python=3.9.19
conda activate countgd
cd CountGD
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
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
* Download the pretrained countgd weight.
  [Google Drive link (1.2 GB)](https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI/view?usp=sharing)

## CountGD Train
setput custome_dataset
```
{
  "train": [
    {
      "root": "../dataset/output_dataset/train/images",
      "anno": "../dataset/output_dataset/train/annotations.jsonl",
      "label_map": "../dataset/output_dataset/label.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "../dataset/output_dataset/valid/images",
      "anno": "../dataset/output_dataset/valid/annotation.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
```

train
```
python main.py --output_dir ./gdino_train -c config/cfg_fsc147_vit_b_odvg.py --datasets custome_data/custome_dataset.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased 
```
## CountBench

See [here](https://github.com/niki-amini-naieni/CountGD/issues/6)

## Citation
If you use our research in your project, please cite our paper.

```
@InProceedings{AminiNaieni24,
  author = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  year = "2024",
}
```

### Acknowledgements

This repository is based on the [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino/tree/main) and uses code from the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO). If you have any questions about our code implementation, please contact us at [niki.amini-naieni@eng.ox.ac.uk](mailto:niki.amini-naieni@eng.ox.ac.uk).
