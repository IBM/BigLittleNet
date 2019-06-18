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

Please refer the instruction there to prepare the ImageNet dataset.

### Training

Training a bL-ResNeXt-101 (64×4d) (α = 2, β = 4) model with two GPUs (0, 1) and saving logfile the `LOGDIR` folder
```
python3 imagenet-train.py --data /path/to/folder -d 101 --basewidth 4 \
--cardinality 64 --backbone_net blresnext --alpha 2 --beta 4 \
--lr_scheduler cosine --logdir LOGDIR --gpu 0,1
```
