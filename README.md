# ml-demo

This is for encoding image and processing image labels.

txt files contains bounded box that are coordinates that are used to draw a box around the object in the image - this is a defect classification dataset.

Notebook has raw images and scattered around txt files - how do we link the image to the txt file?

This notebook extracts image to a snowflake table as well as the data from the text file. Table maps the image to the text file.

Allows for batch processing of images and text files.

After we read images, combine with a pytorch pre-trained model 

Image processing + image training 

Add ~30 - 40 lines of code to add 


First four coordinates are `x min` `y min` `x max` `y max` and last number is the type of defect. For example, disconnect defect, dot defect, etc.

```text
466 441 493 470 3
454 300 493 396 2
331 248 364 283 4
221 314 253 350 4
151 149 182 175 5
492 28 525 55 6
424 24 461 53 6
250 341 278 370 6
539 259 592 316 1
89 469 127 497 5
```

```shell
conda install snowflake-ml-python
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/ddemo/miniconda3

  added / updated specs:
    - snowflake-ml-python


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _py-xgboost-mutex-2.0      |            cpu_0          13 KB
    absl-py-1.4.0              |  py312hca03da5_0         218 KB
    aiobotocore-2.19.0         |  py312hca03da5_0         161 KB
    aiohappyeyeballs-2.4.4     |  py312hca03da5_0          28 KB
    ...
    yarl-1.18.0                |  py312h80987f9_0         147 KB
    ------------------------------------------------------------
                                           Total:       116.1 MB

The following NEW packages will be INSTALLED:

  _py-xgboost-mutex  pkgs/main/osx-arm64::_py-xgboost-mutex-2.0-cpu_0 
  absl-py            pkgs/main/osx-arm64::absl-py-1.4.0-py312hca03da5_0 
  aiobotocore        pkgs/main/osx-arm64::aiobotocore-2.19.0-py312hca03da5_0 
  aiohappyeyeballs   pkgs/main/osx-arm64::aiohappyeyeballs-2.4.4-py312hca03da5_0 
  aiohttp            pkgs/main/osx-arm64::aiohttp-3.11.10-py312h80987f9_0 
  aioitertools       pkgs/main/noarch::aioitertools-0.7.1-pyhd3eb1b0_0 
  aiosignal          pkgs/main/noarch::aiosignal-1.2.0-pyhd3eb1b0_0 
  ...
  yarl               pkgs/main/osx-arm64::yarl-1.18.0-py312h80987f9_0 

The following packages will be DOWNGRADED:

  cloudpickle                         3.0.0-py312hca03da5_0 --> 2.2.1-py312hca03da5_0 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                                                                                                    
Preparing transaction: done                                                                                                                                         
Verifying transaction: done                                                                                                                                         
Executing transaction: done                                                                                                                                         
```

```shell
conda install snowflake-snowpark-python
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/ddemo/miniconda3

  added / updated specs:
    - snowflake-snowpark-python


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    snowflake-snowpark-python-1.30.0|py312hca03da5_100         2.8 MB
    ------------------------------------------------------------
                                           Total:         2.8 MB

The following packages will be UPDATED:

  snowflake-snowpar~               1.29.1-py312hca03da5_100 --> 1.30.0-py312hca03da5_100 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                                                                                                    
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

```shell
(base) ~/Documents/dev/github git:[main]
gh repo clone Charmve/Surface-Defect-Detection
Cloning into 'Surface-Defect-Detection'...
remote: Enumerating objects: 62346, done.
remote: Counting objects: 100% (367/367), done.
remote: Compressing objects: 100% (172/172), done.
remote: Total 62346 (delta 223), reused 313 (delta 187), pack-reused 61979 (from 1)
Receiving objects: 100% (62346/62346), 228.75 MiB | 13.37 MiB/s, done.
Resolving deltas: 100% (54548/54548), done.
Updating files: 100% (62675/62675), done.
```

(base) ~/Documents/dev/github/Surface-Defect-Detection/DeepPCB/PCBData git:[master]

```shell
(base) ~/Documents/dev/github/Surface-Defect-Detection/DeepPCB/PCBData git:[master]
ls
group00041      group12100      group13000      group44000      group77000      group92000      trainval.txt
group12000      group12300      group20085      group50600      group90100      test.txt
```


```shell
python ./load_files.py
```
