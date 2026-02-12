# A Perception CNN for Facial Expression Recognition
## Abstract
Convolutional neural networks (CNNs) can automatically learn data patterns to express face images for facial expression recognition (FER). However, they may ignore effect of facial segmentation of FER. In this paper, we propose a perception CNN for FER as well as PCNN. Firstly, PCNN can use five parallel networks to simultaneously learn local facial features based on eyes, cheeks and mouth to realize the sensitive capture of the subtle changes in FER. Secondly, we utilize a multi-domain interaction mechanism to register and fuse between local sense organ features and global facial structural features to better express face images for FER. Finally, we design a two-phase loss function to restrict accuracy of obtained sense information and reconstructed face images to guarantee performance of obtained PCNN in FER. Experimental results show that our PCNN achieves superior results on several lab and real-world FER benchmarks: CK+, JAFFE, FER2013, FERPlus, RAF-DB and Occlusion and Pose Variant Dataset.
## Diectory strcture
    .
    ├── ...
    ├── dataset
    │   ├── rafdb
    │   │   ├── train
    │   │   └── test
    │   └── ...
    ├── models
    │   └── resnet18_msceleb.pth
    ├── experiment
    │   ├── rafdb
    │   │   └── rafdb.pth
    │   └── ...
    ├── checkpoints
    ├── logs
    └── ...
## Train
Ms-celeb-1m pretrained model, our PCNN model weights and a tiny demo dataset (selected from raf-db) are all available on [GoogleDrive](https://drive.google.com/drive/folders/1st0sETk5Jw0Qs6o4qcAKPn5EWAJJR_vc?usp=sharing)
```shell
python train.py
```
options are available in train.py 
## Test
```shell
python val.py
```
options are available in val.py.  
`val_complex.py` is the script for evaluation of complexity.  
`val_cross.py` is the script for evalaution on occlusion and pose variant datasets.  
## Citation
```bibtex
@article{tian_perception_2025,
  author={Tian, Chunwei and Xie, Jingyuan and Li, Lingjun and Zuo, Wangmeng and Zhang, Yanning and Zhang, David},
  journal={IEEE Transactions on Image Processing}, 
  title={A Perception CNN for Facial Expression Recognition}, 
  year={2025},
  volume={34},
  number={},
  pages={8101 - 8113},
  keywords={Facial expression recognition;sense information;multi-domain interaction;perception network},
  doi={10.1109/TIP.2025.3637715}
}
```
