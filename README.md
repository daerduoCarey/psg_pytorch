Implement [PointSetGen](https://arxiv.org/abs/1612.00603) with ImageNet-pretrained ResNet50 image encoder and FC/FC-UpConv Decoder

Changes:
  - Support both view-centric and shape-centric training (shape-centric achieves better for sure)
  - Support both Chamfer-distance and Earth-mover distance as loss (EMD is slower but performs a little bit better)
  - Training against 10,000 ground-truth points increases the performance trained on 1K/2K ones (this is similar to recent SDF-based methods where usually >10K query points are sampled)

To use, first compile `cd` and `emd` losses, and then run

      bash train.sh

To download the data, please click [here](). Note that this is following the PartNet data splits. You need to switch to the ones used in the other papers.

Code tested on Ubuntu 16.04, Cuda 9.0, Python 3.6.5, PyTorch 1.1.0.

This code uses Blender v2.79 for rendering 3D content for visualization. Please install blender and set it in your system environment path.

Licence: MIT

