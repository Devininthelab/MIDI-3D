# MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation

## 🏠 [Project Page](https://huanngzh.github.io/MIDI-Page/) | [Paper](https://arxiv.org/abs/2412.03558) | [Model](https://huggingface.co/VAST-AI/MIDI-3D) | [Dataset](https://huggingface.co/datasets/huanngzh/3D-Front) | [Online Demo](https://huggingface.co/spaces/VAST-AI/MIDI-3D)

![teaser](assets/doc/teaser.png)

MIDI is a 3D generative model for single image to compositional 3D scene generation. Unlike existing methods that rely on reconstruction or retrieval techniques or recent approaches that employ multi-stage object-by-object generation, MIDI extends pre-trained image-to-3D object generation models to multi-instance diffusion models, enabling the simultaneous generation of multiple high-quality 3D instances with accurate spatial relationships and high generalizability.

## 🌟 Features

* **High Quality:** It produces diverse 3D scenes at high quality with intricate shape.
* **High Generalizability:** It generalizes to real image and stylized image inputs although trained only on synthetic data.
* **High Efficiency:** It generates 3D scenes from segmented instance images, without lengthy steps or time-consuming per-scene optimization.

## 🔥 Updates

* [2025-05] 🌟🌟🌟 Release **textured** 3D scene generation from a single image. Check [it](#textured-3d-scene-generation) out!
* [2025-04] Release [dataset](https://huggingface.co/datasets/huanngzh/3D-Front) and evaluation code.
* [2025-03] Release model weights, gradio demo, inference scripts of MIDI-3D.

## 🔨 Installation

Clone the repo first:

```Bash
git clone https://github.com/VAST-AI-Research/MIDI-3D.git
cd MIDI-3D
```

(Optional) Create a fresh conda env:

```Bash
conda create -n midi python=3.10
conda activate midi
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt
```

For evaluation, you should follow [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install `pytorch3d` package.

## 💡 Usage

The following running scripts will automatically download model weights from [VAST-AI/MIDI-3D](https://huggingface.co/VAST-AI/MIDI-3D) to local directory `pretrained_weights/MIDI-3D`.

### Launch Demo

```Bash
python gradio_demo.py
```

**Important!!** Please check out our instructional video!

https://github.com/user-attachments/assets/814c046e-f5c3-47cf-bb56-60154be8374c

**The web demo is also available on [Hugging Face Spaces](https://huggingface.co/spaces/VAST-AI/MIDI-3D)!**

### Inference Scripts

If running MIDI with command lines, you need to obtain the segmentation map of the scene image firstly. We provide a script to run [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) in `scripts/grounding_sam.py`. The following example command will produce a segmentation map in the `./segmentation.png`.

```Bash
python -m scripts.grounding_sam --image assets/example_data/Cartoon-Style/04_rgb.png --labels lamp sofa table dog --output ./
```

Then you can run MIDI with the rgb image and segmentation map, using our provided inference script `scripts/inference_midi.py`. The following command will save the generated 3D scene `output.glb` in the output dir.

```Bash
python -m scripts.inference_midi --rgb assets/example_data/Cartoon-Style/00_rgb.png --seg assets/example_data/Cartoon-Style/00_seg.png --output-dir "./"
```

**Important!!!**

* We recommend using the [interactive demo](#launch-demo) to get a segmentation map of moderate granularity.
* If instances in your image are too close to the image border, please add `--do-image-padding` to the running scripts of MIDI.

### Textured 3D Scene Generation

> It may require about 40G of video memory. PRs to optimize VRAM requirements are welcome!

You should firstly install [MV-Adapter](https://github.com/huanngzh/MV-Adapter):

```Bash
pip install git+https://github.com/huanngzh/MV-Adapter
```

Then one script to create textured 3d scene!

```Bash
python -m scripts.image_to_textured_scene --rgb_image assets/example_data/Cartoon-Style/01_rgb.png --seg_image assets/example_data/Cartoon-Style/01_seg.png --seed 42 --output output
```

## 📊 Dataset

Our processed [3D-Front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset can be downloaded from [3D-Front (MIDI)](https://huggingface.co/datasets/huanngzh/3D-Front). Please follow the [dataset card](https://huggingface.co/datasets/huanngzh/3D-Front/blob/main/README.md) to organize the dataset:

```Bash
data/3d-front
├── 3D-FRONT-RENDER # rendered views
│   ├── 0a8d471a-2587-458a-9214-586e003e9cf9 # house
│   │   ├── Hallway-1213 # room
│   │   ...
├── 3D-FRONT-SCENE # 3d models (glb)
│   ├── 0a8d471a-2587-458a-9214-586e003e9cf9 # house
│   │   ├── Hallway-1213 # room
│   │   │   ├── Table_e9b6f54f-1d29-47bf-ba38-db51856d3aa5_1.glb # object
│   │   │   ...
├── 3D-FRONT-SURFACE # point cloud (npy)
│   ├── 0a8d471a-2587-458a-9214-586e003e9cf9 # house
│   │   ├── Hallway-1213 # room
│   │   │   ├── Table_e9b6f54f-1d29-47bf-ba38-db51856d3aa5_1.npy # object
│   │   │   ...
├── valid_room_ids.json # scene list
├── valid_furniture_ids.json # object list
├── midi_room_ids.json # scene list (subset used in midi)
└── midi_furniture_ids.json # object list (subset used in midi)
```

Note that we use **the last 1,000 rooms** in `midi_room_ids.json` as the **testset**, and the others as training set.

## Evaluation

> The key code can be found in `test_step` of `midi/systems/system_midi.py` and `midi/utils/metrics.py`.

Before evaluation, download dataset and pre-trained model weights firstly. A simple script to download weights is provided:

```Bash
from huggingface_hub import snapshot_download

REPO_ID = "VAST-AI/MIDI-3D"
local_dir = "pretrained_weights/MIDI-3D"
snapshot_download(repo_id=REPO_ID, local_dir=local_dir)
```

Place the dataset in `data/3d-front` directory, and then you can run the following script to evaluate MIDI on 3D-Front dataset. It will save the results in `outputs/image2scene/test-3dfront/save`, including generated 3D scenes and metrics.

```Bash
python launch.py --config configs/test/3dfront.yaml --test --gpu 0, # --gpu 0,1,2,3,4,5,6,7,
```

If the dataset is not organized in `data/3d-front`, you should modify the config `configs/test/3dfront.yaml`, filling in the correct dataset path.

## Citation

```
@article{huang2024midi,
  title={MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation},
  author={Huang, Zehuan and Guo, Yuanchen and An, Xingqiao and Yang, Yunhan and Li, Yangguang and Zou, Zixin and Liang, Ding and Liu, Xihui and Cao, Yanpei and Sheng, Lu},
  journal={arXiv preprint arXiv:2412.03558},
  year={2024}
}
```
