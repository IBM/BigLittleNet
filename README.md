# BigLittleNet-pytorch

This repository holds the codes and models for the papers.

Chun-Fu (Richard) Chen, Quanfu Fan, Neil Mallinar, Tom Sercu and Rogerio Feris
[Big-Little Net: An Efficient Multi-Scale Feature Representation for Visual and Speech Recognition](https://openreview.net/pdf?id=HJMHpjC9Ym)

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{
    chen2018biglittle,
    title={{Big-Little Net: An Efficient Multi-Scale Feature Representation for Visual and Speech Recognition}},
    author={Chun-Fu (Richard) Chen and Quanfu Fan and Neil Mallinar and Tom Sercu and Rogerio Feris},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=HJMHpjC9Ym},
}
```

## Dependent library
1. pytorch >= 1.0.0
2. tensorboard_logger
3. tqdm

Or install requirement via:

```
pip3 install -r requirement.txt
```

## Usage

The training script is mostly borrow from the imagenet example of [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet) with modifications.

Please refer the instructions there to prepare the ImageNet dataset.

### Training

Training a bL-ResNeXt-101 (64×4d) (α = 2, β = 4) model with two GPUs (0, 1) and saving logfile the `LOGDIR` folder
```
python3 imagenet-train.py --data /path/to/folder -d 101 --basewidth 4 \
--cardinality 64 --backbone_net blresnext --alpha 2 --beta 4 \
--lr_scheduler cosine --logdir LOGDIR --gpu 0,1
```

### Test

After download the models, put in the `pretrained` folder.
Evaluating the bL-ResNeXt-101 (64×4d) (α = 2, β = 4) model with two GPUs.
```
python3 imagenet-train.py --data /path/to/folder -d 101 --basewidth 4 \
--cardinality 64 --backbone_net blresnext --alpha 2 --beta 4 --evaluate \
--gpu 0,1 --pretrained
```

Please feel free to raise issue if you encounter issue when using the pretrained models.


### Results and Models


After the submission, we re-train our models on PyTorch with the same setting described in the paper.

Performance of Big-Little Net models (evaluating on a single 224x224 image.)

|   Model     | Top-1 Error     | FLOPs (10^9) |
|-------------|-----------------|-------|
|[bLResNet-50 (α = 2, β = 4)](https://ibm.box.com/v/blresnet-50-a2-b4)|22.41%|2.85|
|[bLResNet-101 (α = 2, β = 4)](https://ibm.box.com/v/blresnet-101-a2-b4)|21.34%|3.89|
|[bLResNeXt-50 (32x4d) (α = 2, β = 4)](https://ibm.box.com/v/blresnext-50-32x4d-a2-b4)|21.62%|3.03|
|[bLResNeXt-101 (32x4d) (α = 2, β = 4)](https://ibm.box.com/v/blresnext-101-32x4d-a2-b4)|20.87%|4.08|
|[bLResNeXt-101 (64x4d) (α = 2, β = 4)](https://ibm.box.com/v/blresnext-101-64x4d-a2-b4)|20.34%|7.97|
|[bLSEResNeXt-50 (32x4d) (α = 2, β = 4)](https://ibm.box.com/v/blseresnext-50-32x4d-a2-b4)|21.44%|3.03|
|[bLSEResNeXt-101 (32x4d) (α = 2, β = 4)](https://ibm.box.com/v/blseresnext-101-32x4d-a2-b4)|21.04%|4.08|