# Learning To Count Everything mượn xài
![image](learn2countEverything.png)

This is the official implementation of the following CVPR 2021 paper:

```
Learning To Count Everything
Viresh Ranjan, Udbhav Sharma, Thu Nguyen and Minh Hoai
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
```
Link to arxiv preprint: https://arxiv.org/pdf/2104.08391.pdf

Short presentation video

[![Short Presentation](https://img.youtube.com/vi/kdpNq5V2NuA/hqdefault.jpg)](https://youtu.be/kdpNq5V2NuA)
## Dataset download 
Images can be downloaded from here: https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing

Precomputed density maps can be found here: https://archive.org/details/FSC147-GT

Place the unzipped image directory and density map directory inside the data directory.

## Installation with Conda

conda create -n fscount python=3.7 -y

conda activate fscount

python -m pip install matplotlib opencv-python notebook tqdm

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch


## Quick demo

Provide the input image and also provide the bounding boxes of exemplar objects using a text file:

``` bash
python demo.py --input-image 198.jpg --bbox-file 198.txt -m result/FamNet.pth
```

## Evaluation
We are providing our pretrained FamNet model, and the evaluation code can be used without the training.
### Testing on validation split without adaptation
```bash 
python test.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ -m result/FamNet.pth
```
### Testing on val split with adaptation
```bash 
python test.py -dp /PATH/TO/YOUR/DATASET/ -m result/FamNet.pth
```


## Training 
``` bash
python train.py -dp /path/to/dataset -o /path/to/output -ep 100     --gpu 0
```

## Citation

If you find the code useful, please cite:
```
@inproceedings{m_Ranjan-etal-CVPR21,
  author = {Viresh Ranjan and Udbhav Sharma and Thu Nguyen and Minh Hoai},
  title = {Learning To Count Everything},
  year = {2021},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```


