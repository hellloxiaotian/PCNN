# A Perception CNN for Facial Expression Recognition
## Abstract
Convolutional neural networks (CNNs) can automatically learn data patterns to express face images for facial expression recognition (FER). However, they may ignore effect of facial segmentation of FER. In this paper, we propose a perception CNN for FER as well as PCNN. Firstly, PCNN can use five parallel networks to simultaneously learn local facial features based on eyes, cheeks and mouth to realize the sensitive capture of the subtle changes in FER. Secondly, we utilize a multi-domain interaction mechanism to register and fuse between local sense organ features and global facial structural features to better express face images for FER. Finally, we design a two-phase loss function to restrict accuracy of obtained sense information and reconstructed face images to guarantee performance of obtained PCNN in FER. Experimental results show that our PCNN achieves superior results on several lab and real-world FER benchmarks: CK+, JAFFE, FER2013, FERPlus, RAF-DB and Occlusion and Pose Variant Dataset. Its code is available at https://github.com/hellloxiaotian/PCNN.
## Train
Ms-celeb-1m pretrained model, our model weights, datasets are all available on [GooleDrive](https://drive.google.com/drive/folders/1st0sETk5Jw0Qs6o4qcAKPn5EWAJJR_vc?usp=sharing)
~~~shell
python train.py
~~~
options are available in train.py 
## Test
~~~
python val.py
~~~
options are available in val.py
