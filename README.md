# cdc

This is a PyTorch implementation of the paper ["Weakly supervised histopathological image representation
learning based on contrastive dynamic clustering"](). 
```angular2html
@article{liweakly,
  title={Weakly supervised histopathological image representation learning based on contrastive dynamic clustering},
  author={Li, Jun and Jiang, Zhiguo and Zheng, Yushan and Zhang, Haopeng and Shi, Jun and Hu, Dingyi and Luo, Wei and Jiang, Zhongmin and Xue, Chenghai}
}
```
Our code is modified from repository [simsiam](https://github.com/facebookresearch/simsiam).

### Data Preparation
This code use "train.txt" to store the path and pseudo-label of images. An example of "train.txt" file is described as follows:
```angular2html
<path>                         <pseudo-label>
[path to slide1]/0000_0000.jpg 0
[path to slide1]/0000_0001.jpg 0
...
[path to slide2]/0000_0000.jpg 1
...
```
Note: we assign the pseudo-label for the patches from a WSI as the same of
the WSI.

### Training
Firstly, training the [BYOL](https://arxiv.org/abs/2006.07733) for initializing the patch features.
```angular2html
python main_byol.py \
  -a resnet50 \
  --dist-url 'tcp://192.168.0.1:10002' --multiprocessing-distributed --world-size 1 --rank 0\
  --fix-pred-lr \
  [your train.txt file folders]
```

Secondly, fine-tuning the encoder with our cdc module.
```angular2html
python main_cdc.py \
  -a resnet50 \
  --dist-url 'tcp://192.168.0.1:10002' --multiprocessing-distributed --world-size 1 --rank 0\
  --pretrained './checkpoint-byol/checkpoint_0099.pth.tar' \
  [your train.txt file folders]
```
