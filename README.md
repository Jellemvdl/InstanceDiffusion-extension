# A Deep Dive Into "InstanceDiffusion: Instance-level Control for Image Generation"

--- 

### A. Stuger, A. Ibrahimi, L. Becker, R. van Emmerik, J. van der Lee

---

This project aims to reproduce and add an extension to the paper of ["InstanceDiffusion: Instance-level Control for Image Generation"](https://arxiv.org/abs/2402.03290) by Wang et al. (2024). Please refer to our [blogpost](blogpost.md) for detailed information on the implementation of our reproduction and extension of the [InstanceDiffusion model](https://github.com/frank-xwang/InstanceDiffusion).  

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 2.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
- OpenCV ≥ 4.6 is needed by demo and visualization.

### Environment Setup
```bash
conda create --name instdiff python=3.8 -y
conda activate instdiff

pip install -r requirements.txt
```

## Inference Demos

## Evaluation
For evaluation, the ["MSCOCO"](hhttps://cocodataset.org/#download)dataset is used. To evaluate first the MSCOCO dataset was downloaded and stored in the dataset folder. Ensuring the data was organized we stored it as followed: 

```setup
coco/
  annotations/
    instances_val2017.json
  images/
    val2017/
      000000000139.jpg
      000000000285.jpg
      ...
```

Moreover, the customized ["instances_val2017.json"](https://drive.google.com/file/d/1sYpb7jRZJyBJYPFHyjxosIDaiQhkrEhU/view) file needed to be downloaded. This resizes all images to 512x512 and adjusts the corresponding masks/boxes accordingly.

###  Evaluating different location formats for Boxes and Instance Masks

To reproduce the results for evaluating different location formats as input when generating images. This command was used, with different `--test_config` files for boxes and instance masks:

```setup
CUDA_VISIBLE_DEVICES=0 python eval_local.py \
    --job_index 0 \
    --num_jobs 1 \
    --use_captions \
    --save_dir "eval-cocoval17" \
    --ckpt_path pretrained/instancediffusion_sd15.pth \
    --test_config configs/test_mask.yaml \
    --test_dataset cocoval17 \
    --mis 0.36 \
    --alpha 1.0

pip install ultralytics
mv datasets/coco/images/val2017 datasets/coco/images/val2017-official
ln -s generation_samples/eval-cocoval17 datasets/coco/images/val2017
yolo val segment model=yolov8m-seg.pt data=coco.yaml device=0
```

###  Evaluating PiM for Scribble-/Point-based Image Generation

To reproduce the PiM(Points in Mask) results for scribble-/point-based image generation. This command was used, with the same `--test_config` file for both:

```setup
python eval_local.py \
    --job_index 0 \
    --num_jobs 1 \
    --use_captions \
    --save_dir "eval-cocoval17-point" \
    --ckpt_path pretrained/instancediffusion_sd15.pth \
    --test_config configs/test_point.yaml \
    --test_dataset cocoval17 \
    --mis 0.36 \
    --alpha 1.0

pip install ultralytics
mv datasets/coco/images/val2017 datasets/coco/images/val2017-official
ln -s generation_samples/eval-cocoval17-point datasets/coco/images/val2017
yolo val segment model=yolov8m-seg.pt data=coco.yaml device=0

# Please indicate the file path for predictions.json generated in the previous step
python eval/eval_pim.py --pred_json /path/to/predictions.json

###  Evaluating Attribute Binding for colors and texture

To reproduce the attribute binding results for colors and texture. This command was used:

```setup
test_attribute="colors" # colors, textures
CUDA_VISIBLE_DEVICES=0 python eval_local.py \
    --job_index 0 \
    --num_jobs 1 \
    --use_captions \
    --save_dir "eval-cocoval17-colors" \
    --ckpt_path pretrained/instancediffusion_sd15.pth \
    --test_config configs/test_mask.yaml \
    --test_dataset cocoval17 \
    --mis 0.36 \
    --alpha 1.0
    --add_random_${test_attribute}

# Eval instance-level CLIP score and attribute binding performance
python eval/eval_attribute_binding.py --folder eval-cocoval17-colors --test_random_colors
```

In order to evaluate the texture attribute binding performance of InstanceDiffusion, we changed the `test_attribute` to `textures` and substituted `--test_random_textures` for `--test_random_colors`.

## Results
