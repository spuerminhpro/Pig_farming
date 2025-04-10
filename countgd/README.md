# CountGD: Multi-Modal Open World Counting Model

## Overview
CountGD is a multi-modal open-world counting model designed for object detection and counting tasks. It integrates multiple deep learning models, including GroundingDINO and Segment Anything, to provide accurate and efficient counting solutions.

## For training code:

[Train CountGD Repository](https://github.com/spuerminhpro/Pig_farming/tree/main/train_countgd)


## Table of Contents
- [Preparation](#preparation)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Docker Deployment](#docker-deployment)
- [License](#license)

---

## Preparation
Before proceeding, ensure you have the necessary dependencies installed on your system.

### Install GCC
The project requires GCC for compiling certain components. The tested versions include GCC 11.3 and 11.4. Install GCC and other development tools using:

```bash
sudo apt update
sudo apt install build-essential
```

---

## Installation
This repo run on CUDA 12.8
### Clone the Repository
Clone the repository using the following command:

```bash
git clone https://github.com/spuerminhpro/Pig_farming.git
cd Pig_farming/train_countgd
```

### Set Up the Environment
Create and activate a Conda environment for CountGD:

```bash
conda create -n countgd python=3.11 -y
conda activate countgd
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
```

### Build and Install Dependencies
Navigate to the required directory and install the necessary dependencies:

```
cd models/GroundingDINO/ops
python setup.py build
pip install .
python test.py # should result in 6 lines of * True
pip install git+https://github.com/facebookresearch/segment-anything.git
cd ../../../
```

---

## Usage
To run the model with a pre-trained checkpoint, execute:

```bash
python app.py --pretrain_model_path /path/to/best_checkpoint
```

---

for public the link
modify the app.py
```
interface.launch(share=True)
```



## Evaluation
To evaluate the model on a test dataset, open and execute `evaluate_test.ipynb`.

---

## Docker Deployment
To run the application using Docker, use the following command:

```bash
docker run -it \
    --name countgd \
    -p 7860:7860 \
    --platform=linux/amd64 \
    --gpus all \
    registry.hf.space/nikigoli-countgd:latest \
    python app.py --pretrain_model_path /path/to/best_checkpoint
```

---

## License
This project is open-source and available under the [MIT License](LICENSE).

For any issues or contributions, feel free to submit a pull request or open an issue in the repository.

