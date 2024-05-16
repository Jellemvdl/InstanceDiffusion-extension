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

In order to run the Inference Demos of InstanceDiffusion locally, we provide [`src/scripts/inference.py/`](src/scripts/inference.py/) and different json files in [`src/lib/instancediff/`](src/lib/instancediff/), specifying text prompts and location conditions for generating specific images. In order to run these demos, please install the pretrained InstanceDiffusion from [Hugging Face](https://huggingface.co/xudongw/InstanceDiffusion/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1Jm3bsBmq5sHBnaN5DemRUqNR0d4cVzqG?usp=sharing) and [SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt), place them under the [`src/lib/instancediff/pretrained`](src/lib/instancediff/pretrained) folder. Run the inference demos using the notebooks in [`demos/`](demos)

### Image Generation Using Single Points

InstanceDiffusion supports image generation by using points, with each point representing an instance, along with corresponding instance captions.

<p align="center">
  <img src="src/data/images/InstDiff-points.png" width=95%>
</p>

### Iterative Image Generation
  InstanceDiffusion supports iterative image generation with minimal changes to pre-generated instances and the overall scene. By using the same initial noise and image caption, InstanceDiffusion can selectively introduce new instances, replace existing ones, reposition instances, or adjust their sizes by modifying the bounding boxes.

https://github.com/frank-xwang/InstanceDiffusion/assets/58996472/b161455a-6b21-4607-a59d-3a6dd19edab1

### Add more/different demo examples 
...

### Extension/GPT4 demo

*Add visualisation of gpt bounding box demo*

## Evaluation
For evaluation, the [MSCOCO](https://cocodataset.org/#download) dataset is used. To evaluate first the MSCOCO dataset was downloaded and stored in the dataset folder. Ensuring the data was organized we stored it as followed: 

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

Moreover, the customized [instances_val2017.json](https://drive.google.com/file/d/1sYpb7jRZJyBJYPFHyjxosIDaiQhkrEhU/view) file needed to be downloaded. This resizes all images to 512x512 and adjusts the corresponding masks/boxes accordingly.

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

To reproduce the PiM(Points in Mask) results for scribble-/point-based image generation. This command was used, with different `--test_config` files for both:

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
``` 

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

### Reproduction

As part of our reproduction study, we succesfully replicated the YOLO results achieved by the original authors for different location formats as input when generating images for Boxes and Instance masks: 
<table align="center">
  <tr align="center">
      <th align="left">Method</th>
      <th>AP<sub>box</sub></th>
      <th>AP<sub>box</sub><sup>50</sup></th>
      <th>AR<sub>box</sub></th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>38.8</td>
    <td>55.4</td>
    <td>52.9</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>49.9</td>
    <td>66.8</td>
    <td>68.6</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+11.1</td>
    <td>+11.4</td>
    <td>+15.7</td>
  </tr>
  <tr align="left">
    <td colspan=4><b>Table 1.</b> Evaluating different location formats as input when generating images of reproduction experiments Boxes.</td>
  </tr>
</table>

<table align="center">
  <tr align="center">
      <th align="left">Method</th>
      <th>AP<sub>mask</sub></th>
      <th>AP<sub>mask</sub><sup>50</sup></th>
      <th>AR<sub>mask</sub></th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>27.1</td>
    <td>50.0</td>
    <td>38.1</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>40.8</td>
    <td>63.5</td>
    <td>56.0</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+13.7</td>
    <td>+13.5</td>
    <td>+17.9</td>
  </tr>
  <tr align="left">
    <td colspan=4><b>Table 2.</b> Evaluating different location formats as input when generating images of reproduction experiments Instance Masks.</td>
  </tr>
</table>

In the same manner we reproduced the PiM values for scribble-/point-based image generation:**NOT YET ACTUALLY**
<table align="center">
  <tr align="center">
      <th align="left" rowspan="2">Method</th>
      <th colspan="1">Points</th>
      <th colspan="1">Scribble</th>
  </tr>
  <tr align="center">
      <th>PiM</th>
      <th>PiM</th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>81.1</td>
    <td>72.4</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>..</td>
    <td>..</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td>+..</td>
    <td>+..</td>
  </tr>
  <tr align="left">
    <td colspan=3><b>Table 3.</b> Evaluating different location formats as input when generating images of reproduction experiments for points and scribbles.</td>
  </tr>
</table>

Moreover, succesfully replicated the attribute binding results for colors and texture:

<table align="center">
  <tr align="center">
      <th align="left" rowspan="2">Methods</th>
      <th colspan="2">Color</th>
      <th colspan="2">Texture</th>
  </tr>
  <tr align="center">
      <th>A<sub>cc</sub><sup>color</sup></th>
      <th>CLIP<sub>local</sub></th>
      <th>A<sub>cc</sub><sup>texture</sup></th>
      <th>CLIP<sub>local</sub></th>
  </tr>
  <tr align="center">
    <td align="left">InstanceDiffusion</td>
    <td>54.4</td>
    <td>0.250</td>
    <td>26.8</td>
    <td>0.225</td>
  </tr>
  <tr align="center">
    <td align="left">Our Reproduction</td>
    <td>53.3</td>
    <td>0.248</td>
    <td>26.9</td>
    <td>0.226</td>
  </tr>
  <tr align="center">
    <td align="left">Difference</td>
    <td style="color:red;">-1.1</td>
    <td style="color:red;">-0.002</td>
    <td style="color:green;">+0.1</td>
    <td style="color:green;">+0.001</td>
  </tr>
    <tr align="left">
    <td colspan=7><b>Table 4.</b> Attribute binding reproduction results for color and texture.</td>
  </tr>  
</table>

