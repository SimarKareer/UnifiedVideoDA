# VideoDA
Domain adaptation for semantic segmentation using video!

This repo is built off of mmsegmentation, with the [MIC repo](https://github.com/lhoyer/MIC/tree/master)

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
- `tools/aggregate_flows/flow/my_utils.py`
- `tools/aggregate_flows/flow/util_flow.py`

**VideoDA techinques**
- Video Discriminator:
    - `mmseg/models/uda/dacsAdvseg.py`
- PL Refinement:
    - `mmseg/models/uda/dacs.py`
- ACCEL + Consistent Mixup:
    - `mmseg/models/segmentors/accel_hrda_encoder_decoder.py`
    - `mmseg/models/utils/dacs_transforms.py`

**Dataset and Model Configurations**
- `configs/_base_/datasets/*`
- `configs/mic/*`

**Experiment Scripts**
- `tools/experiments/*`



## Reproducing Results

All experiments conducted in the paper have corresponding scripts for reproducability inside the repository. Due to certain configurations, we have separated our code into 2 branches: (1) `discrim` (2) `accel`.

The `discrim` branch will have all experiments that predict on single-images (no accel architecture) and will have scripts for reproducing results on viper and synthia for the baseline (HRDA + MIC) with different backbones, Pseudo-label refinement (with forward/backward flow compatibility), and video discriminator.

The `accel` branch will have all experiments reported that predict on consecutive frames (ACCEL architecture) for viper. This brnach will also include the techniques needed for consistent mixup, which we have coupled with the ACCEL architecture for our experiments.

All experiment scrips are located in `tools/experiments/*`, with scripts being separated by the different shifts and VideoDA techniques.


## Accel Branch Experiments

### Viper -> CityscapesSeq

<ins>**Table 4: Combining existing Video-DA methods with HRDA**</ins>

DLV2 Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + Accel + Consis Mixup| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup.sh` |
| HRDA + Accel + Consis Mixup + Video Discrim| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_video_discrim.sh` |
| (MOM) HRDA + Accel + Consis Mixup + PL Refine Consis Filter| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_consis_filter.sh` |
| (TPS)HRDA + Accel + Consis Mixup + PL Refine Warp Frame | `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_warp_frame.sh` |
| (DAVSN) HRDA + Accel + Consis Mixup + Video Discrim + PL Refine Max confidence| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_video_discrim_consis_filter.sh` |
| HRDA + Accel + Consis Mixup + Video Discrim + PL Refine Consis Filter| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_video_discrim_consis_filter.sh` |


