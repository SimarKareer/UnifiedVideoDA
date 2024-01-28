# Unified VideoDA
Codebase for "We're Not Using Videos Effectively: An Updated Domain Adaptive Video Segmentation Baseline"

This repo is built off of mmsegmentation, with the [MIC repo](https://github.com/lhoyer/MIC/tree/master)

## Installation
Modification of these [instructions](https://github.com/lhoyer/MIC/tree/master/seg).

1. `conda create -n mic python=3.8.5`
2. `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`
3. Install my mmcv from scratch.  [full instructions if necessary](https://mmcv.readthedocs.io/en/latest/get_started/build.html)
    - `git submodule update --recursive` will pull my mmcv submodule
    - Simply run `MMCV_WITH_OPS=1 pip install -e . -v` inside the `submodules/mmcv` directory
4. `pip install -e .` inside mmseg root dir

## Dataset Setup (Cityscapes-Seq, Synthia-Seq, Viper)

Please download the following datasets, which will be used in Video-DAS experiments.  Download to `mmseg/datasets` with the following structure and key folders
```
datasets/
├── cityscapes-seq
│   ├── gtFine % gtFine_trainvaltest.zip
│   ├── leftImg8bit_sequence % gtFine_trainvaltest.zip
└── VIPER
    ├── val
    ├── test
    ├── train
        └── img % Images: Frames: *0, *1, *[2-9]; Sequences: 01-77; Format: jpg
        └── cls % Semantic Class Labels: Frames: *0, *1, *[2-9]; Sequences: 01-77; Format: png
└── SynthiaSeq
    └── SYNTHIA-SEQS-04-DAWN
        └── RGB
        └── GT
```

Download Links:
* [Cityscapes-Seq](https://www.cityscapes-dataset.com/)
* [Viper](https://playing-for-benchmarks.org/download/): 
* [Synthia-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar) 

After downlaoding all datasets, we must generate sample class statistics on our source datasets (Viper, Synthia-Seq) and convert class labels into Cityscapes-Seq classes.

For both Viper and Synthia-Seq, perform the following:
```bash
python tools/convert_datasets/viper.py datasets/viper --gt-dir train/cls/

python tools/convert_datasets/synthiaSeq.py datasets/SynthiaSeq/SYNTHIA-SEQS-04-DAWN --gt-dir GT/LABELS/Stereo_Left/Omni_F
```

## Dataset Setup (BDDVid)
We introduce support for a new target domain dataset derived from BDD10k. BDD10k has a series of 10,000 driving images across a variety of conditions.  Of these 10,000 images, we identify 3,429 with valid corresponding video clips in the BDD100k dataset, making this subset suitable for Video-DAS. We refer to this subset as BDDVid. Next, we split these 3,429 images into 2,999 train samples and 430 evaluation samples. In BDD10k, the labeled frame is generally the 10th second in the 40-second clip, but not always. To mitigate this, we ultimately only evaluate images in BDD10k that perfectly correspond with the segmentation annotation, while at training time we use frames directly extracted from BDD100k video clips. 

The following instructions below will give detail in how to set up BDDVid Dataset.

1. **Download Segmentation Labels for BDD10k (https://bdd-data.berkeley.edu/portal.html#download) images.**

2. **Download all BDD100k video parts: **

    ```bash
    cd datasets/BDDVid/setup/download
    python download.py --file_name bdd100k_video_links.txt
    ```

    Note: Make sure to specify the correct output directory in `download.py` for where you want the video zips to be stored.

3. **Unzip all video files**

    ```bash
    cd ../unzip
    python unzip.py
    ```

    Note: Make sure to specify the directory for where the video zips are stored and the output directory for where files should be unzipped in `unzip.py`

4. **Unpack each video sequence and extract the corresponding frame**

   ```bash
   cd ../unpack_video
   ```

    Create a text file with paths to each video unzipped. Refer to `video_path_train.txt` and `video_path_val.txt` as an example.

    ```bash
    python unpack_video.py
    ```

    Note: You will run the script twice, based on the split we are unpacking for (train or val). Edit the `split` varibale to specify train or val, and the `file_path` variable, which refers to the list of all video paths for the given split.

    Also, note that through experimentation and analysis, we determined frame 307 in the videos is the closest to the images in the BDD10k dataset. We deal with the possible slight label mismatch problem in later steps to counter this issue.

5. **Download [BDD10k](https://bdd-data.berkeley.edu/portal.html#download) ("10k Images") and its labels ("Segmentation" tab), and unzip them.**

6. **Copy Segmentation labels for train and val in BDDVid**

    ```bash
    cd ../bdd10k_img_labels
    python grab_labels.py
    ```

    Note: Run this 2 times for each split (train, val). Edit the `orig_dataset` with the path to the original BDD10k  dataset train split, which was downlaoded in step 5.

7. **Fix Image-Label Mismatch**

    We will be creating 2 new folders to deal with the image-label mismatch at frame 307 described in step (4).

    (1) `train_orig_10k`
        - same as train, but the 307 frame is from the original BDD10k dataset. Use this directory for supervised BDD jobs

    (2) `val_orig_10k`
        - same as val, but the 307 frame is from the original BDD10k dataset. *ALWAYS* use this split, as we want to compute validation over the actual image and label. 

    ```bash
    python get_orig_images.py
    ```

    Note: Run this 2 times for each split (train, val). Edit the `orig_dataset` with the path to the original BDD10k dataset train spit, which was downloaded in step 5.


BDDVid is finally setup! For UDA jobs, use the `train` and `val_orig_10k` split. For supervised jobs with BDDVid, use `train_orig_10k` and `val_orig_10k`.


## Dataset Setup (Optical Flow)

A number of our methods rely on optical flow between successive frames, thus for each dataset, we generated flows using [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official). We have hosted all our generated flows for each dataset on [Hugging Face](https://huggingface.co/datasets/hoffman-lab/Unified-VideoDA-Generated-Flows).

Simply run 
```bash
git lfs install
git clone https://huggingface.co/datasets/hoffman-lab/Unified-VideoDA-Generated-Flows
```
This will produce the following file tree
```
Unified-VideoDA-Generated-Flows/
├── SynthiaSeq_Flows
│   └── frame_dist_1
│       └── im
│           ├── synthiaSeq_im_backward_flow.tar.gz
│           ├── synthiaSeq_im_forward_flow.tar.gz
├── BDDVid_Flows
│   └── frame_dist_2
│       ├── imtk
│       │   └── bddvid_imtk_backward_flow.tar.gz
│       └── im
│           └── bddvid_im_backward_flow.tar.gz
├── Viper_Flows
│   └── frame_dist_1
│       ├── imtk
│       │   └── viper_imtk_backward_flow.tar.gz
│       └── im
│           ├── viper_im_forward_flow.tar.gz
│           └── viper_im_backward_flow.tar.gz
├── CityscapesSeq_Flows
│   └── frame_dist_1
│       ├── imtk
│       │   ├── csSeq_imtk_forward_flow.tar.gz
│       │   └── csSeq_imtk_backward_flow.tar.gz
│       └── im
│           ├── csSeq_im_backward_flow.tar.gz
│           └── csSeq_im_forward_flow.tar.gz
```
Finally unpack each tar file.  For instance:
```
cd Unified-VideoDA-Generated-Flows/SynthiaSeq_Flows/frame_dist_1/im
tar -xvzf synthiaSeq_im_backward_flow.tar.gz.tar.gz
```

## Reproducing Experiments
See [`./experiments.md`](./experiments.md) for commands to run any experiment in the paper.  The HRDA baseline can be run via `python tools/train.py configs/mic/viperHR2bddHR_mic_hrda.py --launcher=slurm --l-warp-lambda=0.0 --l-mix-lambda=1.0 --seed 1 --deterministic --work-dir=./work_dirs/<dirname> --nowandb True`

## Key Contributions to mmsegmentation Repo
We have made a number of key contributions to this open source mmsegmentation repo to support video domain adaptative segmentation experiments for future researchers to build off of. 

Firstly, we consolidated the HRDA + MIC works into the  mmsegmentation repository. By adding the SOTA ImageDA work into this repository,researchers have the capability of easily switching between models, backbones, segmentation heads, and architectures for experimentation and ablation studies.

We added key datasets for the VideoDA benchmark (ViperSeq -> CityscapesSeq, SynthiaSeq -> CityscapesSeq) to mmsegmentation, along with our own constructed shift (ViperSeq -> BDDVid, SynthiaSeq -> BDDVid) , and allowed for the capability of loading consecutive images along with the corresponding optical flow based on a frame distance specified. This enables researchers to easily start work on VideoDA related problems or benchmark current ImageDA appraoches on this setting.

In additon, we provide implementations of common VideoDA techniques such as Video Discriminators, ACCEL architectures + consistent mixup, and a variety of pseudo-label refinement strategies.

All experiments we report in our paper have been made avaiabile in the repository, with each experiment's corresponding bash script to help with reproducability. The next section covers these scripts.

The following files are where key changes were made:

**VideoDA Dataset Support**
- `mmseg/datasets/viperSeq.py`
- `mmseg/datasets/cityscapesSeq.py`
- `mmseg/datasets/SynthiaSeq.py`
- `mmseg/datasets/SynthiaSeq.py`
- `mmseg/datasets/bddSeq.py`

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

## Citation
```
@inproceedings{kareer2024NotUsingVideosCorrectly
    title={We're are Not Using Videos Effectively: An Updated Video Domain Adaptation Baseline},
    author={Simar Kareer, Vivek Vijaykumar, Harsh Maheshwari, Prithvi Chattopadhyay, Judy Hoffman, Viraj Prabhu},
    booktitle={Transactions on Machine Learning Research (TMLR)},<br></br>
    &emsp;&emsp;year={2024}
}
```



