# 3DPmmGAN 
Implementation and data for **A deep-learning-based porous media microstructure quantitative characterization and reconstruction method**

## Environment
 - Nvidia series graphics card, and a Windows or Linux system computer
 - [Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
 - [TensorFlow 1.10](https://img.shields.io/badge/tensorflow-1.10-green.svg?style=plastic)
 - [cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
 - In addition you will need to have installed `h5py` and `tifffile`
```bash
pip install h5py
pip install tifffile
```

## Training
All models were trained on `NVIDIA GeForce RTX 2080Ti` GPUs.  
Use the 96 resolution version, according to the recommended settings we have arranged, the training time on the two graphics cards does not exceed 24 hours.
To create a data set from the original sample, please follow the steps below: 
1. Run 'create_training_images.py' to obtain sub-volume samples (hdf5).
2. Run 'dataset_h5_tool.py' 
```bash
python dataset_h5_tool.py create_from_hdf5 output input
```

##  Results
 - We have provided the results used to create our publication in "analysis".
 - We have provided an interesting model--'Aggregation' Model. Using the example direction vector we provided, you can observe the same change trend for different structures.
 - We have shown some cases of this model in the 'paper' folder.

## Additional material can be found on Google Drive:
 - 'Aggregation' Dataset: https://drive.google.com/file/d/1NV3p-rgDNSCExAMcwf84ZsZ7w0EXXrOS/view?usp=sharing
 - 'Aggregation' Model: https://drive.google.com/file/d/1qhuhaIXd51P318OVlnBdJOjy3wCrQ_4W/view?usp=sharing
 - A pre-trained model of 96 resolution: https://drive.google.com/file/d/1JiAwHQouM5Omlmubsw98FBXXwMw02llh/view?usp=sharing

## Acknowledgement
The code used for our research is based on [StyleGAN](https://github.com/NVlabs/stylegan) and [PorousMediaGAN](https://github.com/LukasMosser/PorousMediaGan).
The original samples used in this study are from [the collection of Imperial College London](https://www.imperial.ac.uk/earth-science/research/research-groups/perm/research/pore-scale-modelling-and-imaging/micro-ct-images-and-networks).