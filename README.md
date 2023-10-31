# Resnet-Tensorflow-Pytorch
Code for resnet and its depth variants. Code is in both Tensorflow and Pytorch

```
usage: main.py [-h] [--model {tf,torch}]
               [--depth {9,18,20,32,44,50,56,101,152,200}]
               [--dataset {imagenet,cifar10}]

Create Resnet model in Tensorflow or Pytorch

optional arguments:
  -h, --help            show this help message and exit
  --model {tf,torch}    Model will be created on Tensorflow, Pytorch (default:
                        tf)
  --depth {9,18,20,32,44,50,56,101,152,200}
                        Resnet model depth (default: 50)
  --dataset {imagenet,cifar10}
                        Imagenet depth (Default) : 9|18|50|101|152|200
                        CIFAR10: 20|32|44|56
```

To create model in Tensorflow:

For Imagenet dataset:

```
python3 main.py --model tf --depth 50 --dataset imagenet
```

For CIFAR-10 and CIFAR-100 dataset:

```
python3 main.py --model tf --depth 44 --dataset cifar10
```