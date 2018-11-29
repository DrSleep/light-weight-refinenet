# Light-Weight RefineNet (in PyTorch)

This repository provides official models from the paper `Light-Weight RefineNet for Real-Time Semantic Segmentation`, available [here](http://bmvc2018.org/contents/papers/0494.pdf)

```
Light-Weight RefineNet for Real-Time Semantic Segmentation
Vladimir Nekrasov, Chunhua Shen, Ian Reid
In BMVC 2018
```

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

## More to come

Once time permits, more things will be added to this repository:

* NASNet-Mobile
* CityScapes' models
* Full training pipeline example
* Evaluation scripts

## More projects to check out

Our most recent work on real-time joint semantic segmentation and depth estimation is built on top of Light-Weight RefineNet with MobileNet-v2. Check out the paper [here](https://arxiv.org/abs/1809.04766); the models are soon to be released!

## License

For academic usage, this project is licensed under the 2-clause BSD License - see the [LICENSE](LICENSE) file for details. For commercial usage, please contact the authors.

## Acknowledgments

* University of Adelaide and Australian Centre for Robotic Vision (ACRV) for making this project happen
* HPC Phoenix cluster at the University of Adelaide for making the training of the models possible
* PyTorch developers
* Google Colab
* Yerba mate tea
