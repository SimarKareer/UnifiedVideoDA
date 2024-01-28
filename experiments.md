
## Reproducing Experiment Results

All experiments conducted in the paper have corresponding scripts for reproducability inside the repository. 

All experiment scripts are located in `tools/experiments/*`, with scripts being separated by the different shifts and VideoDA techniques.

### Viper -> CityscapesSeq
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
| Source Only| `./tools/experiments/viper_csSeq/baselines/viper_source_hrda_dlv2.sh` |
| HRDA | `./tools/experiments/viper_csSeq/baselines/viper_csseq_mic_hrda_dlv2.sh` |
| (TPS) HRDA + Accel + Consis Mixup + PL Refine Warp Frame | `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_warp_frame.sh` |
| (DAVSN) HRDA + Accel + Consis Mixup + Video Discrim + PL Refine Max confidence| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_video_discrim_consis_filter.sh` |
| (UDA-VSS) HRDA + Accel + Video Discrim + PL Refine Consis Filter| `tools/experiments/viper_csSeq/video_discrim/viper_csseq_hrda_dlv2_video_discrim_consis.sh` |
| (MOM) HRDA + Accel + Consis Mixup + PL Refine Consis Filter| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_consis_filter.sh` |
| HRDA + Video Discrim. | `./tools/experiments/viper_csSeq/video_discrim/viper_csseq_hrda_dlv2_video_discrim.sh` |
| HRDA + Accel + Consis Mixup| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup.sh` |
| HRDA + PL refine Consis Filter| `tools/experiments/viper_csSeq/pl_refinement/consis/viper_csseq_mic_hrda_dlv2_consis.sh` |
| HRDA + Accel + Consis Mixup + Video Discrim| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_video_discrim.sh` |
| HRDA + Accel + Consis Mixup + Video Discrim + PL Refine Consis Filter| `tools/experiments/viper_csSeq/accel/viper_csseq_hrda_dlv2_accel_consis_mixup_video_discrim_consis_filter.sh` |
| Target Only| `./tools/experiments/csSeq/supervised/csSeq_supervised_hrda_dlv2.sh` |

<br>
<br>


Note: [For Tables 5-8] To train with forward or backwards flow, edit `FRAME_OFFSET`  (positive values = forward, negative values = backwards) in `configs/_base_/datasets/uda_viper_CSSeq.py` along with `cs_train_flow_dir` and `cs_val_flow_dir`.

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


### Other Shifts

SynthiaSeq -> CityscapesSeq, SynthiaSeq -> BDDVid, ViperSeq --> BDDVid experiment scripts follow directory structure as the Viper Experiments. You can find all relevant experiments reported in the paper at `tools/experiments/*`.
