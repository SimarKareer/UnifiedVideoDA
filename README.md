# VideoDA
Domain adaptation for semantic segmentation using video!

This repo is built off of mmseg.  I used the [MIC repo](https://github.com/lhoyer/MIC/tree/master)

## Installation
Modification of these [instructions](https://github.com/lhoyer/MIC/tree/master/seg).

1. `conda create -n mic python=3.8.5`
2. `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`
3. Install my mmcv from scratch.  [full instructions if necessary](https://mmcv.readthedocs.io/en/latest/get_started/build.html)
    - `git submodule update --recursive` will pull my mmcv submodule
    - Simply run `MMCV_WITH_OPS=1 pip install -e . -v` inside the `submodules/mmcv` directory
4. `pip install -e .` inside mmseg root dir


## Key Contributions to mmsegmentation Repo
We have made a number of key contributions to this open source mmsegmentation repo to support video domain adaptative segmentation experiments for future researchers to build off of. 

Firstly, we consolidated the HRDA + MIC works into the  mmsegmentation repository. By adding the SOTA ImageDA work into this repository,researchers have the capability of easily switching between models, backbones, segmentation heads, and architectures for experimentation and ablation studies.

We added key datasets for the VideoDA benchmark (ViperSeq -> CityscapesSeq, SynthiaSeq -> CityscapesSeq) to mmsegmentation, and allowed for the capability of loading consecutive images along with the corresponding optical flow based on a frame distance specified. This enables researchers to easily start work on VideoDA related problems or benchmark current ImageDA appraoches on this setting.

In additon, we provide implementations of common VideoDA techniques such as Video Discriminators, ACCEL architectures + consistent mixup, and a variety of pseudo-label refinement strategies.

All experiments we report in our paper have been made avaiabile in the repository, with each experiment's corresponding bash script to help with reproducability. The next section covers these scripts.

The following files are where key changes were made:

**VideoDA Dataset Support**
- `mmseg/datasets/viperSeq.py`
- `mmseg/datasets/cityscapesSeq.py`
- `mmseg/datasets/SynthiaSeq.py`
- `mmseg/datasets/SynthiaSeq.py`

**Consecutive Frame/Optical Flow Support**
- `mmseg/datasets/seqUtils.py`
-[TODO: insert flow utils]
- loading.py changes

**VideoDA techinques**
- video discrim file
- other branch files
- pl refinement (dacs)


**Configurations**
- `configs/\_base\_//datasets/`
- `configs/mic/*`

**Experiment Scripts**
- `tools/experiments`

## Reproducing Results

All exper


