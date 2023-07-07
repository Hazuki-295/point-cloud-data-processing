# point-cloud-processing

## Getting started

Create a new conda environment
```zsh
conda env create -f environment.yml
```
Activate the conda environment
```zsh
conda activate point-cloud-library
```

(macOS) For dynamic linker/loader to find `libomp` when importing `open3d`, set the dynamic library search path
```zsh
conda env config vars set DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```