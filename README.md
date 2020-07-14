# Light-Weight RefineNet (in PyTorch)

[![Build Status](https://api.travis-ci.com/DrSleep/light-weight-refinenet.svg?branch=master)](https://travis-ci.com/DrSleep/light-weight-refinenet)
[![Test Coverage](https://codecov.io/gh/DrSleep/light-weight-refinenet/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/gh/DrSleep/light-weight-refinenet?branch=master)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/light-weight-refinenet-for-real-time-semantic/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=light-weight-refinenet-for-real-time-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/light-weight-refinenet-for-real-time-semantic/real-time-semantic-segmentation-on-nyu-depth-1)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-nyu-depth-1?p=light-weight-refinenet-for-real-time-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/light-weight-refinenet-for-real-time-semantic/semantic-segmentation-on-pascal-voc-2012)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012?p=light-weight-refinenet-for-real-time-semantic)


This repository provides official models from the paper `Light-Weight RefineNet for Real-Time Semantic Segmentation`, available [here](http://bmvc2018.org/contents/papers/0494.pdf)

```
Light-Weight RefineNet for Real-Time Semantic Segmentation
Vladimir Nekrasov, Chunhua Shen, Ian Reid
In BMVC 2018
```

## UPDATES

### 14, July, 2020: 
1. New weights of Light-Weight RefineNet with the ResNet-50 backbone trained on COCO+BSD+VOC via the code in [src_v2/](https://github.com/DrSleep/light-weight-refinenet/tree/master/src_v2) have been uploaded. The model shows *82.04*% mean iou on the validation set in the single-scale regime, and *83.41*% mean iou on the test set with multi-scale and horizontal flipping ([per-class test results](http://host.robots.ox.ac.uk/anonymous/UBQIQJ.html)).
2. New weights of Light-Weight RefineNet with the MobileNet-v2 backbone trained on COCO+BSD+VOC via the code in [src_v2/](https://github.com/DrSleep/light-weight-refinenet/tree/master/src_v2) have been uploaded. The model shows *78.30*% mean iou on the validation set in the single-scale regime, and *80.28*% mean iou on the test set with multi-scale and horizontal flipping ([per-class test results](http://host.robots.ox.ac.uk/anonymous/R7Q4IV.html)).


### 5, June, 2020: a new version of the code has been pushed. It currently resides in [src_v2/](https://github.com/DrSleep/light-weight-refinenet/tree/master/src_v2). The code now closely interacts with [densetorch](https://github.com/DrSleep/DenseTorch) and supports transformations from [albumentations](https://github.com/albumentations-team/albumentations), while also supporting [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html) datasets. Three training examples are provided in [train/](https://github.com/DrSleep/light-weight-refinenet/tree/master/train):
1. [train_v2_nyu.sh](https://github.com/DrSleep/light-weight-refinenet/blob/master/train/train_v2_nyu.sh) is analogous to [nyu.sh](https://github.com/DrSleep/light-weight-refinenet/blob/master/train/nyu.sh), trains Light-Weight-RefineNet-50 on NYU, achieving ~42.4% mean IoU on the validation set (no TTA).
2. [train_v2_nyu_albumentations.sh](https://github.com/DrSleep/light-weight-refinenet/blob/master/train/train_v2_nyu_albumentations.sh) uses transformations from the albumentations package, achieving ~42.5% mean IoU on the validation set (no TTA).
3. [train_v2_sbd_voc.sh](https://github.com/DrSleep/light-weight-refinenet/blob/master/train/train_v2_sbd_voc.sh) trains Light-Weight-RefineNet-50 on SBD (5623 training images) and VOC (1464 training images) datasets from torchvision with transformations from the albumentations package; achieves ~76% mean IoU on the validation set with no TTA (1449 validation images).

If you want to train the network on your own dataset, specify the arguments (see the available options in [src_v2/arguments.py](https://github.com/DrSleep/light-weight-refinenet/blob/master/src_v2/arguments.py)) and provide implementation of your dataset in [src_v2/data.py](https://github.com/DrSleep/light-weight-refinenet/blob/master/src_v2/data.py#L283) if it is not supported by either [densetorch](https://github.com/DrSleep/DenseTorch) or [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html).

## Getting Started

For flawless reproduction of our results, the Ubuntu OS is recommended. The models have been tested using Python 2.7 and Python 3.6.

### Dependencies

```
pip, pip3
torch>=0.4.0
```
To install required Python packages, please run `pip install -r requirements.txt` (Python2), or `pip3 install -r requirements3.txt` (Python3) - use the flag `-u` for local installation.
The given examples can be run with, or without GPU.

## Running examples

For the ease of reproduction, we have embedded all our examples inside Jupyter notebooks. One can either download them from this repository and proceed working with them on his/her local machine/server, or can resort to online version supported by the Google Colab service.

### Jupyter Notebooks [Local]

If all the installation steps have been smoothly executed, you can proceed with running any of the notebooks provided in the `examples/notebooks` folder.
To start the Jupyter Notebook server, on your local machine run `jupyter notebook`. This will open a web page inside your browser. If it did not open automatically, find the port number from the command's output and paste it into your browser manually.
After that, navigate to the repository folder and choose any of the examples given. 

The number of FLOPs and runtime are measured on 625x468 inputs using a single GTX1080Ti, mean IoU is given on corresponding validation sets with a single scale input.

|Models|[PASCAL VOC](./examples/notebooks/VOC.ipynb) | [Person-Part](./examples/notebooks/PersonPart.ipynb)  |[PASCAL Context](./examples/notebooks/Context.ipynb)| [NYUv2, 40](./examples/notebooks/NYU.ipynb) | Params, M | FLOPs, B| Runtime, ms
| -------- |:-------------:| -----:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| RF-LW-ResNet-50      | 78.5 | 64.9 | - | 41.7 | 27 | 33 | **19.56±0.29** 
| RF-LW-ResNet-101      | 80.3      | 66.7  | 45.1 | 43.6 | 46 | 52 | 27.16±0.19
| RF-LW-ResNet-152 | **82.1**      | **67.6** | **45.8** | **44.4** | 62 | 71 | 35.82±0.23
| RF-LW-MobileNet-v2 | 76.2      |   - | - | - | **3.3** | **9.3** | - 

Inside the notebook, one can try out their own images, write loops to iterate over videos / whole datasets / streams (e.g., from webcam). Feel free to contribute your cool use cases of the notebooks!

### Colab Notebooks [Web]

If you do not want to be involved in any hassle regarding the setup of the Jupyter Notebook server, you can proceed by using the same examples inside the Google colab environment - with free GPUs available! 

* [PASCAL Context](https://colab.research.google.com/drive/1WI61ZoXu9Wh8lMYmmrweq0DfrmHkiIqY)
* [PASCAL Person-Part](https://colab.research.google.com/drive/1EUYkocpVMDTusCvLPa9EW-pG8Wie6esh)
* [PASCAL VOC](https://colab.research.google.com/drive/1J5P8yCOrjpeDcEeF5Haj_MQeb7SGF5vi)
* [NYUv2-40](https://colab.research.google.com/drive/1S5wvuukFM6GTLbj8VxZFdkFn2jhdhiES)

## Training scripts

We provide training scripts to get you started on the [NYUv2-40](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset. The methodology slightly differs from the one described in the paper and leads to better and more stable results (at least, on NYU).

In particular, here we i) start with a lower learning rate (as we initialise weights using default PyTorch's intiialisation instead of normal(0.01)), ii) add more aggressive augmentation (random scale between 0.5 and 2.0), and iii) pad each image inside the batch to a fixed crop size (instead of resizing all of them). The training process is divided into 3 stages: after each the optimisers are re-created with the learning rates halved. All the training is done using a single GTX1080Ti GPU card.
Additional experiments with this new methodology on the other datasets (and with the MobileNet-v2 backbone) are under way, and relevant scripts will be provided once available. Please also note that the training scripts were written in Python 3.6.

To start training on NYU:

0. If not already done, download the dataset from [here](https://cloudstor.aarnet.edu.au/plus/s/OZqPy3RnmbCkOWh). Note that the white borders in all the images were already cropped.
1. Build the helper code for calculating mean IoU written in Cython. For that, execute the following `python src/setup.py build_ext --build-lib=./src/`.
2. Make sure to provide the correct paths to the dataset images either by modifying `src/config.py` or `train/nyu.sh`
3. Run `./train/nyu.sh`. On a single 1080Ti, the training takes around 3-6 hours (ResNet-50 - ResNet-152, correspondingly).

If you want to train the networks using your dataset, you would need to modify the following:

1. Add files with paths to your images and segmentation masks. The paths can either be relative or absolute - additional flags `TRAIN_DIR` and `VAL_DIR` in `src/config.py` can be used to prepend the relative paths. It is up to you to decide how to encode the segmentation masks - in the NYU example, the masks are encoded without a colourmap, i.e., with a single digit (label) per 2-D location;
2. Make sure to adapt the implementation of the NYUDataset for your case in `src/datasets.py`: in particular, pay attention to how the images and masks are being read from the files;
3. Modify `src/config.py` for your needs - do not forget about changing the number of classes (`NUM_CLASSES`);
4. Finally, run your code - see `train/nyu.sh` for example. 


## More to come

Once time permits, more things will be added to this repository:

* NASNet-Mobile
* CityScapes' models
* ~~Full training pipeline example~~
* ~~Evaluation scripts~~ (`src/train.py` provides the flag `--evaluate`)

## More projects to check out

1. Our most recent work on real-time joint semantic segmentation and depth estimation is built on top of Light-Weight RefineNet with MobileNet-v2. Check out the paper [here](https://arxiv.org/abs/1809.04766); the models are available [here](https://github.com/DrSleep/multi-task-refinenet)!
2. RefineNet-101 trained on PASCAL VOC is available [here](https://github.com/DrSleep/refinenet-pytorch)

## License

For academic usage, this project is licensed under the 2-clause BSD License - see the [LICENSE](LICENSE) file for details. For commercial usage, please contact the authors.

## Acknowledgments

* University of Adelaide and Australian Centre for Robotic Vision (ACRV) for making this project happen
* HPC Phoenix cluster at the University of Adelaide for making the training of the models possible
* PyTorch developers
* Google Colab
* Yerba mate tea
