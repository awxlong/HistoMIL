# HistoMIL
![HistoMIL](https://github.com/secrierlab/HistoMIL/blob/main/logo.png)

### Original Author: Shi Pan, UCL Genetics Institute

HistoMIL is a Python package for handling histopathology whole-slide images using multiple instance learning (MIL) techniques. With HistoMIL, you can create MIL datasets, train, cross-validate, evaluate MIL models, make MIL predictions on new slide images and perform interpretability analysis:

![Pipeline](https://github.com/awxlong/HistoMIL/blob/jupyter/figs/pipeline.png)

The scripts submitted to the Sun Grid Engine scheduler executing each of the above steps in the pipeline can be found at:  https://github.com/awxlong/scripts_g0_arrest

### Implementation details:

HistoMIL is written in Pytorch Lightning, which provides the following benefits:
- mixed precision training
- gradient accumulation over patches
- model checkpointing for resuming crashed experiments

We implement the following MIL algorithms:
1. [TransMIL](https://github.com/szc19990412/TransMIL)
2. TransMILRegression: TransMIL outputting regression scores instead of classification probabilities
3. TransMILMultimodal: TransMIL with multimodal fusion of clinical features
4. [ABMIL](https://github.com/axanderssonuu/ABMIL-ACC) 
5. [DSMIL](https://github.com/binli123/dsmil-wsi)
6. [Transformer](https://github.com/peng-lab/HistoBistro) 
7. TransformerRegression 
8. TransformerMultimodal
9. [AttentionMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)
10. [CAMIL](https://github.com/olgarithmics/ICLR_CAMIL)
11. [DTFD_MIL](https://github.com/hrzhang1123/DTFD-MIL)
12. [GraphTransformer](https://github.com/vkola-lab/tmi2022)
13. [CLAM](https://github.com/mahmoodlab/CLAM)
14. A hybrid DTFD-MIL-TransMIL, where attention over pseudo-bags is replaced with TransMIL's Nystrom attention 


## Installing HistoMIL

To use HistoMIL, you first need to create a conda environment with the required dependencies.

### create env with pre-defined file
You can do this by importing the env.yml file provided in this repository:

### linux user pre-requisites
1. Create conda env
```bash
conda create -n HistoMIL python=3.9
```
This will create a new environment named histomil, which you can activate with:

```bash
conda activate HistoMIL
```

### windows user pre-requisites

Windows (10+)
1. Download OpenSlide binaries from this page. Extract the folder and add bin and lib subdirectories to Windows system path. If you are using a conda environment you can also copy bin and lib subdirectories to [Anaconda Installation Path]/envs/YOUR ENV/Library/.

2. Install OpenJPEG. The easiest way is to install OpenJpeg is through conda using

```bash
conda create -n HistoMIL python=3.9
```
This will create a new environment named histomil, which you can activate with:

```bash
conda activate HistoMIL
```

```bash
C:\> conda install -c conda-forge openjpeg
```

### macOS user pre-requisites
On macOS there are two popular package managers, homebrew and macports.

Homebrew
```bash
brew install openjpeg openslide
```
MacPorts
```bash
port install openjpeg openslide
```

### create env manually 

Then install openslide and pytorch-gpu with following scripts.

```bash
conda install -c conda-forge openslide
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

Next, install the required Python packages with pip:

```bash
pip install -r requirements.txt
```
This will install all the packages listed in requirements.txt, including HistoMIL itself.


## Usage

All of the examples for using HistoMIL are included in the Notebooks folder. You can open and run these Jupyter notebooks to see how to use HistoMIL for different histopathology tasks.

## Contributing

If you find a bug or want to suggest a new feature for HistoMIL, please open a GitHub issue in this repository. Pull requests are also welcome!

## License

HistoMIL is released under the GNU-GPL License. See the LICENSE file for more information.
