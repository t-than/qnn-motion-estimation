# Quaternion neural networks for motion estimation in HEVC

> **Work in progress.** This repository is updated as parts of the project are reviewed.

A quaternion neural network predicts motion vectors that are fed into the motion search of
the HEVC reference encoder (HM). The effect is measured as BD-rate against unmodified HM.

This repository contains the tools to reproduce the measurements:

- building HM, and **HM-mv**: HM-18.0 with a patch (`week2/hm-mv.patch`) that records the
  encoder's motion vectors and takes motion vectors from outside;
- downloading and preparing the test sequences and training data;
- running encodes in parallel, each checked with the reference decoder;
- computing BD-rate (`bdrate.py`, checked against the `bjontegaard` package).

## Requirements

- Linux with a C++ compiler, CMake, git, ffmpeg, 7-Zip (`7z`), unzip and aria2.
  `week1/setup_arch.sh` installs these on Arch Linux and creates the Python environment.
- Python 3 with numpy, scipy, pandas, matplotlib, tqdm, torch, torchvision and bjontegaard.
- An NVIDIA GPU for the network parts.
- About 15 GB of disk for the test sequences and results, plus about 100 GB for the training data.

## Quick start

```bash
cd week1
./setup_arch.sh                          # Arch only; on other systems install the packages above
source .venv/bin/activate
./build_hm.sh                            # clones HM, builds HM-18.0, smoke test
python bdrate.py --selftest              # must print "selftest PASSED"

./download_data.sh small uvg             # test sequences
./download_data.sh chairs sintel         # training data (large)
python prepare_sequences.py              # 33-frame clips -> data/seq/, sequences.json
python prepare_sequences.py --sets uvg --step 2 --tag 60fps

python run_encodes.py --variant anchor                     # unmodified HM, QP 22/27/32/37
python run_encodes.py --variant sr8 --extra="--SearchRange=8"
python bdrate.py results/anchor_lowdelay_P.csv results/sr8_lowdelay_P.csv
```

HM-mv, the patched encoder:

```bash
cd week2
./build_hm_mv.sh                         # HM-18.0 + hm-mv.patch, in a git worktree of week1/HM
python ../week1/run_encodes.py --results-dir results --variant dump \
    --encoder HM-mv/bin/TAppEncoderStatic --decoder HM-mv/bin/TAppDecoderStatic \
    --extra="--MEDumpFile={dir}/me.dump --MVFieldDumpFile={dir}/field.mvf"
python mvfield.py info results/dump/lowdelay_P/foreman_352x288/QP32/field.mvf
```

## Data

The data is not part of this repository. `download_data.sh` fetches it from the original sources;
re-running it resumes interrupted downloads. Please respect each dataset's licence:

| Dataset | Use | Source | Licence |
|---|---|---|---|
| Xiph test media (CIF) | test | media.xiph.org/video/derf | see each sequence |
| UVG 1080p | test | ultravideo.fi | CC BY-NC; cite Mercat et al., MMSys 2020 |
| FlyingChairs | training | lmb.informatik.uni-freiburg.de | research only, no commercial use; cite Dosovitskiy et al., ICCV 2015 |
| MPI-Sintel | training | sintel.is.tue.mpg.de | see the dataset page; cite Butler et al., ECCV 2012 |

## Licence

HM is © ITU/ISO/IEC and contributors under the BSD 3-clause licence; `week2/hm-mv.patch`
modifies HM and is distributed under the same licence. For the rest of this repository, see `LICENSE`.
