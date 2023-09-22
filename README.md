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


## Discrim Branch Experiments

### Viper -> CityscapesSeq

Note: To train with frames t and t+1 on the target dataset (CityscapesSeq), replace the path for the data split for the target dataset in `train.txt` in `configs/_base_/datasets/uda_viper_CSSeq.py`.

<br>

<ins>**Table 2: ImageDA methods on VideoDA benchmarks:**</ins>

Segformer Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + MIC| `./tools/experiments/viper_csSeq/baselines/viper_csseq_mic_hrda.sh` |
| HRDA | `./tools/experiments/viper_csSeq/baselines/viper_csseq_hrda.sh` |
| Target Only| `./tools/experiments/csSeq/supervised/csSeq_supervised_hrda.sh` |
| Source Only | `./tools/experiments/viper_csSeq/baselines/viper_source_hrda.sh` |


DLV2 Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + MIC| `./tools/experiments/viper_csSeq/baselines/viper_csseq_mic_hrda_dlv2.sh` |
| HRDA | `./tools/experiments/viper_csSeq/pl_refinement/consis/viper_csseq_hrda_dlv2_consis.sh` |
| Target Only| `./tools/experiments/csSeq/supervised/csSeq_supervised_hrda_dlv2.sh` |
| Source Only| `./tools/experiments/viper_csSeq/baselines/viper_source_hrda_dlv2.sh` |
<br>

<ins>**Table 3: HRDA + MIC Ablation Study**</ins>

DLV2 Backbone
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA - MRFusion| `./tools/experiments/viper_csSeq/mic_hrda_component_ablation/viper_csseq_hrda_dlv2_no_MRFusion.sh` |
| HRDA - MRFusion - Rare class sampling | `./tools/experiments/viper_csSeq/mic_hrda_component_ablation/viper_csseq_hrda_dlv2_no_MRFusion_no_rcs.sh` |
| HRDA - MRFusion - Rare class sampling - ImgNet feature distance reg| `./tools/experiments/viper_csSeq/mic_hrda_component_ablation/viper_csseq_hrda_dlv2_no_MRFusion_no_rcs_no_imnet.sh` |

<br>

<ins>**Table 4: Combining existing Video-DA methods with HRDA**</ins>


DLV2 Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + Video Discrim. | `./tools/experiments/viper_csSeq/video_discrim/viper_csseq_hrda_dlv2_video_discrim.sh` |
| HRDA + Video Discrim. + PL refine Consis Filter| `./tools/experiments/viper_csSeq/video_discrim/viper_csseq_hrda_dlv2_video_discrim_consis.sh` |

<br>
<br>


Note: [For Tables 5-8] To train with forward or backwards flow, edit `FRAME_OFFSET` (positive value = forward, negative value = backwards) in `configs/_base_/datasets/uda_viper_CSSeq.py` along with `cs_train_flow_dir` and `cs_val_flow_dir`.

<ins>**Table 5: Psuedo-label Refinement on HRDA + MIC, Segformer Backbone**</ins>

Segformer Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + MIC + PL Refine Consis Filter| `./tools/experiments/viper_csSeq/pl_refinement/max_conf/viper_csseq_mic_hrda_max_conf.sh` |
| HRDA + MIC + PL Refine Max Confidence | `./tools/experiments/viper_csSeq/pl_refinement/rare_class_filter/viper_csseq_mic_hrda_rare_class_filter.sh` |
| HRDA + MIC + PL Refine Warp Frame| `./tools/experiments/viper_csSeq/pl_refinement/warp_frame/viper_csseq_mic_hrda_warp_frame.sh` |
| HRDA + MIC + PL Refine Oracle | `./tools/experiments/viper_csSeq/pl_refinement/oracle/viper_csseq_mic_hrda_oracle.sh` |
<br>

<ins>**Table 6: Psuedo-label Refinement on HRDA, Segformer Backbone**</ins>

Segformer Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + PL Refine Consis Filter| `./tools/experiments/viper_csSeq/pl_refinement/consis/viper_csseq_hrda_consis.sh` |
| HRDA + PL Refine Max Confidence | `./tools/experiments/viper_csSeq/pl_refinement/max_conf/viper_csseq_hrda_max_conf.sh` |
| HRDA + PL Refine Warp Frame| `./tools/experiments/viper_csSeq/pl_refinement/warp_frame/viper_csseq_hrda_warp_frame.sh` |
| HRDA + PL Refine Oracle | `./tools/experiments/viper_csSeq/pl_refinement/oracle/viper_csseq_hrda_oracle.sh` |
<br>

<ins>**Table 7: Psuedo-label Refinement on HRDA + MIC, DLV2 Backbone**</ins>

DLV2 Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + MIC + PL Refine Consis Filter| `../tools/experiments/viper_csSeq/pl_refinement/consis/viper_csseq_mic_hrda_dlv2_consis.sh` |
| HRDA + MIC + PL Refine Max Confidence | `./tools/experiments/viper_csSeq/pl_refinement/max_conf/viper_csseq_mic_hrda_dlv2_max_conf.sh` |
| HRDA + MIC + PL Refine Warp Frame| `./tools/experiments/viper_csSeq/pl_refinement/warp_frame/viper_csseq_mic_hrda_dlv2_warp_frame.sh` |
| HRDA + MIC + PL Refine Oracle | `./tools/experiments/viper_csSeq/pl_refinement/oracle/viper_csseq_mic_hrda_dlv2_oracle.sh` |

<br>

<ins>**Table 8: Psuedo-label Refinement on HRDA, DLV2 Backbone**</ins>

Segformer Backbone:
| Experiment | Training Script |
| ------------- | ------------- |
| HRDA + PL Refine Consis Filter| `./tools/experiments/viper_csSeq/pl_refinement/consis/viper_csseq_hrda_dlv2_consis.sh` |
| HRDA + PL Refine Max Confidence | `./tools/experiments/viper_csSeq/pl_refinement/max_conf/viper_csseq_hrda_dlv2_max_conf.sh` |
| HRDA + PL Refine Warp Frame| `./tools/experiments/viper_csSeq/pl_refinement/warp_frame/viper_csseq_hrda_dlv2_warp_frame.sh` |
| HRDA + PL Refine Oracle | `./tools/experiments/viper_csSeq/pl_refinement/oracle/viper_csseq_hrda_dlv2_oracle.sh` |

### SynthiaSeq -> CityscapesSeq

SynthiaSeq experiment scripts follow directory structure as the Viper Experiments. You can find all relevant experiments reported in the paper at `tools/experiments/synthiaSeq_csSeq`.





