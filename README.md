# point-cloud-processing

## Getting started

1. Create a new conda environment:
    ```zsh
    conda env create -f environment.yml
    ```
2. Activate the conda environment:
    ```zsh
    conda activate point-cloud-library
    ```

3. (macOS) For dynamic linker/loader to find `libomp` when importing `open3d`, set the dynamic library search path:
    ```zsh
    conda env config vars set DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
    ```
   Then reactivate the environment:
    ```zsh
    conda deactivate
    conda activate point-cloud-library
    ```