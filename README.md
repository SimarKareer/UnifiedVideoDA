# VideoDA
Domain adaptation for semantic segmentation using video!

This repo is built off of mmseg.  I used the [MIC repo](https://github.com/lhoyer/MIC/tree/master)

## Installation
Modification of these [instructions](https://github.com/lhoyer/MIC/tree/master/seg).

1. `conda create -n mic python=3.8.5`
2. `pip install -r requirements/requirements.txt`
3. Install my mmcv from scratch.  [full instructions if necessary](https://mmcv.readthedocs.io/en/latest/get_started/build.html)
    - `git submodule update --recursive` will pull my mmcv submodule
    - Simply run `MMCV_WITH_OPS=1 pip install -e . -v` inside the `submodules/mmcv` directory
4. `pip install -e .` inside mmseg root dir