---
license: mit
task_categories:
- question-answering
language:
- en
size_categories:
- 1M<n<10M
---
## License & Third-party Terms

**Our work in this repo (code + our annotations/splits)** is released under **MIT License** (see `LICENSE`).  
We **do not** redistribute any original images/depth/meshes/panoramas from third-party datasets.

**Third-party datasets required (not included here):**
- **ScanNet** — obtain from the official website under its Terms of Use (non-commercial research/education).
- **ScanNet++** — obtain from the official website under its Terms of Use (non-commercial research/education).
- **Matterport3D / RXR** — panoramas/environments must be obtained under the Matterport3D Terms of Use.
- **Structured3D** — obtain from the official release under its license/terms.

To reproduce our structure:
1. First **download each dataset from its official source** and accept their terms.
2. Then run our preparation scripts to locally materialize the expected folder layout.
3. This repository contains **only** our code and annotation files; **no third-party originals** are uploaded.

> **If you have any questions or encounter difficulties obtaining the original datasets (e.g., network access issues), please open a GitHub issue at [GitHub issues] or email [contact]. We will do our best to provide technical assistance, but please note you must follow the original datasets' Terms of Use.**

[GitHub issues]: https://github.com/fudan-zvg/spar/issues
[contact]: jiahzhang23@m.fudan.edu.cn

<p align="left">
  <a href="https://github.com/fudan-zvg/spar.git">
    <img alt="GitHub Code" src="https://img.shields.io/badge/Code-spar-black?&logo=github&logoColor=white" />
  </a>
  <a href="https://arxiv.org/abs/xxx">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-spar-red?logo=arxiv" />
  </a>
  <a href="https://fudan-zvg.github.io/spar">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-spar-blue" />
  </a>
</p>


# 📦 Spatial Perception And Reasoning Dataset (SPAR-7M)
> A large-scale vision-language dataset designed for **spatial perception and reasoning**.

**SPAR-7M** contains over **7 million QA pairs** across **33 diverse spatial tasks**, generated from **4,500+ richly annotated 3D indoor scenes**. It supports **single-view**, **multi-view**, and **video-based** image inputs, and features both **perception** and **reasoning**-oriented question types.

This dataset serves as the foundation for [SPAR-Bench](https://huggingface.co/datasets/jasonzhango/SPAR-Bench), and is suitable for **pretraining**, **multitask learning**, and **spatial grounding** research.

This version supports **single-view**, **multi-view**, and **video-based** inputs.

## 📥 Download

We provide **two versions** of the dataset:

| Version          | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `SPAR-7M`        | **QA annotations only** — contains question/answer JSONL files, and splits (no original images are included). |
| `SPAR-7M-RGBD`   | Includes **depths**, **camera intrinsics**, and **pose matrices** for 3D-aware training

> **Important:** Neither version redistributes third-party raw images/meshes/panoramas (e.g., ScanNet, ScanNet++, Matterport3D, Structured3D).  
> To reproduce the full input (images/depths) you must **obtain those datasets from their official sources under their Terms of Use** and then run the scripts we provided to locally materialize the SPAR folder layout expected by our code.


You can download both versions from **Hugging Face**:

```bash
# Download SPAR-7M (default)
huggingface-cli download jasonzhango/SPAR-7M --repo-type dataset

# Download SPAR-7M-RGBD (with depth and camera parameters)
huggingface-cli download jasonzhango/SPAR-7M-RGBD --repo-type dataset
```

These datasets are split into multiple .tar.gz parts due to Hugging Face file size limits. After downloading all parts, run the following to extract:
```
# NOTE: the old pattern `cat spar-*.tar.gz | tar -xvzf -` is deprecated for our current releases.
# We now provide per-dataset archives (annotations only for SPAR-7M), so extract each .tar.gz separately.

# For SPAR-7M (annotations only)
tar -xvzf scannet.tar.gz
tar -xvzf scannetpp.tar.gz
tar -xvzf structured3d.tar.gz
tar -xvzf rxr.tar.gz

# For SPAR-7M-RGBD
cat spar-rgbd-*.tar.gz | tar -xvzf -
```

Alternatively, if Hugging Face is not accessible, you can use the [provided script](https://hf-mirror.com/):
```
wget https://hf-mirror.com/hfd/hfd.sh

chmod a+x hfd.sh

export HF_ENDPOINT=https://hf-mirror.com

./hfd.sh jasonzhango/SPAR-7M --dataset
./hfd.sh jasonzhango/SPAR-7M-RGBD --dataset
```

Prepare local image folders from original third-party datasets

After you download the original third-party datasets (ScanNet, ScanNet++, Matterport3D, Structured3D) and accept their terms, run the corresponding preparation scripts in this repository to materialize the spar/.../images/ layout expected by our code. **The usage instructions are included inside each script.**

Example usages (run on the machine where you placed the original datasets):
```
# 1) Prepare ScanNet layout (example)
SCANNET_ROOT=/path/to/scannet \
OUT_ROOT=./spar/scannet/images \
python jasonzhango/SPAR-7M/prepare_scannet_layout.py --scannet-root "$SCANNET_ROOT" --out-root "$OUT_ROOT" --use-video-idx

# 2) Prepare ScanNet++ layout
SCANNETPP_ROOT=/path/to/scannetpp \
OUT_ROOT=./scannetpp/images \
python jasonzhango/SPAR-7M/prepare_scannetpp_layout.py --scannetpp-root "$SCANNETPP_ROOT" --out-root "$OUT_ROOT" --use-video-idx

# 3) Prepare Structured3D layout
STRUCTURED3D_ROOT=/path/to/Structured3D \
OUT_ROOT=./structured3d/images \
python jasonzhango/SPAR-7M/prepare_structured3d_layout.py --structured3d-root "$STRUCTURED3D_ROOT" --out-root "$OUT_ROOT"

```

- **Important**: these scripts do not download third-party datasets for you — you must obtain ScanNet / ScanNet++ / Matterport3D / Structured3D from their official sources and place them on your machine according to each script's --*-root argument (see script headers for expected source layout).

- Once the scripts finish, the repo-local spar/.../images/ folders will be populated and ready for training/evaluation.

(Thanks to the wonderful GPT for the usage docs 🫶)


The dataset directory structure is:
```
spar/
├── rxr/
├── scannet/
│   ├── images/
│   |   └── scene0000_00/
│   |       ├── image_color/
│   |       ├── video_color/
│   |       ├── image_depth/           # only in SPAR-7M-RGBD
│   |       ├── video_depth/           # only in SPAR-7M-RGBD
│   |       ├── pose/                  # only in SPAR-7M-RGBD
│   |       ├── video_pose/            # only in SPAR-7M-RGBD
│   |       ├── intrinsic/             # only in SPAR-7M-RGBD
│   |       └── video_idx.txt
│   └── qa_jsonl/
│       ├── train/
│       |   ├── depth_prediction_oo/
│       |   |   ├── fill/
│       |   |   |   └── fill_76837.jsonl
│       |   |   ├── select/
│       |   |   └── sentence/
│       |   ├── obj_spatial_relation_oc/
│       |   └── spatial_imagination_oo_mv/
│       └── val/
├── scannetpp/
└── structured3d/
```
Each QA task (e.g., `depth_prediction_oc`, `spatial_relation_oo_mv`, etc.) is organized by **task type**, with subfolders for different **answer formats**:
  - `fill/` — numerical or descriptive answers
  - `select/` — multiple choice
  - `sentence/` — natural language answers

## 📚 Bibtex

If you find this project or dataset helpful, please consider citing our paper:

```bibtex
@article{zhang2025from,
    title={From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D},
    author={Zhang, Jiahui and Chen, Yurui and Xu, Yueming and Huang, Ze and Mei, Jilin and Chen, Junhui and Zhou, Yanpeng and Yuan, Yujie and Cai, Xinyue and Huang, Guowei and Quan, Xingyue and Xu, Hang and Zhang, Li},
    year={2025},
    journal={arXiv preprint arXiv:2503.22976},
}
```

<!-- ## 📄 License

This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You may use, share, modify, and redistribute this dataset **for any purpose**, including commercial use, as long as proper attribution is given.

[Learn more](https://creativecommons.org/licenses/by/4.0/) -->