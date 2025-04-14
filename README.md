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
/Users/dgillis/miniconda3/lib/python3.12/site-packages/conda/base/context.py:201: FutureWarning: Adding 'defaults' to channel list implicitly is deprecated and will be removed in 25.3. 

To remove this warning, please choose a default channel explicitly with conda's regular configuration system, e.g. by adding 'defaults' to the list of channels:

  conda config --add channels defaults

For more information see https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/use-condarc.html

  deprecated.topic(
/Users/dgillis/miniconda3/lib/python3.12/site-packages/conda/base/context.py:201: FutureWarning: Adding 'defaults' to channel list implicitly is deprecated and will be removed in 25.3. 

To remove this warning, please choose a default channel explicitly with conda's regular configuration system, e.g. by adding 'defaults' to the list of channels:

  conda config --add channels defaults

For more information see https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/use-condarc.html

  deprecated.topic(
Channels:
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/dgillis/miniconda3

  added / updated specs:
    - snowflake-ml-python


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _py-xgboost-mutex-2.0      |            cpu_0          13 KB
    absl-py-1.4.0              |  py312hca03da5_0         218 KB
    aiobotocore-2.19.0         |  py312hca03da5_0         161 KB
    aiohappyeyeballs-2.4.4     |  py312hca03da5_0          28 KB
    aiohttp-3.11.10            |  py312h80987f9_0         910 KB
    aioitertools-0.7.1         |     pyhd3eb1b0_0          20 KB
    aiosignal-1.2.0            |     pyhd3eb1b0_0          12 KB
    anyio-4.6.2                |  py312hca03da5_0         259 KB
    arrow-cpp-17.0.0           |       h0b7d223_1         8.1 MB
    attrs-24.3.0               |  py312hca03da5_0         171 KB
    aws-c-auth-0.6.19          |       h80987f9_0          87 KB
    aws-c-cal-0.5.20           |       h80987f9_0          39 KB
    aws-c-common-0.8.5         |       h80987f9_0         186 KB
    aws-c-compression-0.2.16   |       h80987f9_0          18 KB
    aws-c-event-stream-0.2.15  |       h313beb8_0          45 KB
    aws-c-http-0.6.25          |       h80987f9_0         164 KB
    aws-c-io-0.13.10           |       h80987f9_0         133 KB
    aws-c-mqtt-0.7.13          |       h80987f9_0          62 KB
    aws-c-s3-0.1.51            |       h80987f9_0          63 KB
    aws-c-sdkutils-0.1.6       |       h80987f9_0          45 KB
    aws-checksums-0.1.13       |       h80987f9_0          48 KB
    aws-crt-cpp-0.18.16        |       h313beb8_0         203 KB
    aws-sdk-cpp-1.11.212       |       hdd7fb2f_0         3.4 MB
    blas-1.0                   |         openblas          10 KB
    boost-cpp-1.82.0           |       h48ca7d4_2          12 KB
    botocore-1.36.3            |  py312hca03da5_0         8.7 MB
    bottleneck-1.4.2           |  py312ha86b861_0         124 KB
    cachetools-5.5.1           |  py312hca03da5_0          36 KB
    cloudpickle-2.2.1          |  py312hca03da5_0          48 KB
    frozenlist-1.5.0           |  py312h80987f9_0          52 KB
    fsspec-2024.12.0           |  py312hca03da5_0         394 KB
    gflags-2.2.2               |       h313beb8_1         125 KB
    glog-0.5.0                 |       h313beb8_1          92 KB
    importlib_resources-6.4.0  |  py312hca03da5_0          77 KB
    jmespath-1.0.1             |  py312hca03da5_0          48 KB
    joblib-1.4.2               |  py312hca03da5_0         516 KB
    libboost-1.82.0            |       h0bc93f9_2        18.2 MB
    libbrotlicommon-1.0.9      |       h80987f9_9          70 KB
    libbrotlidec-1.0.9         |       h80987f9_9          29 KB
    libbrotlienc-1.0.9         |       h80987f9_9         291 KB
    libevent-2.1.12            |       h02f6b3c_1         391 KB
    libgfortran-5.0.0          |11_3_0_hca03da5_28         142 KB
    libgfortran5-11.3.0        |      h009349e_28         1.0 MB
    libgrpc-1.62.2             |       h62f6fdd_0         5.5 MB
    libopenblas-0.3.21         |       h269037a_0         3.3 MB
    libthrift-0.15.0           |       h73c2103_2         333 KB
    libxgboost-2.1.1           |       h313beb8_0         1.8 MB
    llvm-openmp-14.0.6         |       hc6e5704_0         253 KB
    multidict-6.1.0            |  py312h80987f9_0          54 KB
    numexpr-2.10.1             |  py312h5d9532f_0         181 KB
    numpy-1.26.4               |  py312h7f4fdc5_0          12 KB
    numpy-base-1.26.4          |  py312he047099_0         6.3 MB
    orc-2.0.1                  |       h937ddfc_0         449 KB
    pandas-2.2.3               |  py312hcf29cfe_0        14.6 MB
    propcache-0.3.1            |  py312h80987f9_0          55 KB
    py-xgboost-2.1.1           |  py312hca03da5_0         365 KB
    pyarrow-17.0.0             |  py312h313beb8_1         4.3 MB
    python-tzdata-2023.3       |     pyhd3eb1b0_0         140 KB
    pytimeparse-1.1.8          |  py312hca03da5_0          19 KB
    re2-2022.04.01             |       hc377ac9_0         165 KB
    retrying-1.3.3             |     pyhd3eb1b0_2          14 KB
    s3fs-2024.12.0             |  py312hca03da5_0          83 KB
    scikit-learn-1.5.2         |  py312h313beb8_0         9.6 MB
    scipy-1.15.2               |  py312h14fe6a6_1        22.0 MB
    snappy-1.2.1               |       h313beb8_0          39 KB
    sniffio-1.3.0              |  py312hca03da5_0          18 KB
    snowflake-ml-python-1.8.1  |py312hca03da5_100         1.4 MB
    sqlparse-0.5.2             |  py312hca03da5_0         101 KB
    threadpoolctl-3.5.0        |  py312h989b03a_0          49 KB
    utf8proc-2.6.1             |       h80987f9_1          97 KB
    wrapt-1.17.0               |  py312h80987f9_0          63 KB
    xgboost-2.1.1              |  py312hca03da5_0          12 KB
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
  anyio              pkgs/main/osx-arm64::anyio-4.6.2-py312hca03da5_0 
  arrow-cpp          pkgs/main/osx-arm64::arrow-cpp-17.0.0-h0b7d223_1 
  attrs              pkgs/main/osx-arm64::attrs-24.3.0-py312hca03da5_0 
  aws-c-auth         pkgs/main/osx-arm64::aws-c-auth-0.6.19-h80987f9_0 
  aws-c-cal          pkgs/main/osx-arm64::aws-c-cal-0.5.20-h80987f9_0 
  aws-c-common       pkgs/main/osx-arm64::aws-c-common-0.8.5-h80987f9_0 
  aws-c-compression  pkgs/main/osx-arm64::aws-c-compression-0.2.16-h80987f9_0 
  aws-c-event-stream pkgs/main/osx-arm64::aws-c-event-stream-0.2.15-h313beb8_0 
  aws-c-http         pkgs/main/osx-arm64::aws-c-http-0.6.25-h80987f9_0 
  aws-c-io           pkgs/main/osx-arm64::aws-c-io-0.13.10-h80987f9_0 
  aws-c-mqtt         pkgs/main/osx-arm64::aws-c-mqtt-0.7.13-h80987f9_0 
  aws-c-s3           pkgs/main/osx-arm64::aws-c-s3-0.1.51-h80987f9_0 
  aws-c-sdkutils     pkgs/main/osx-arm64::aws-c-sdkutils-0.1.6-h80987f9_0 
  aws-checksums      pkgs/main/osx-arm64::aws-checksums-0.1.13-h80987f9_0 
  aws-crt-cpp        pkgs/main/osx-arm64::aws-crt-cpp-0.18.16-h313beb8_0 
  aws-sdk-cpp        pkgs/main/osx-arm64::aws-sdk-cpp-1.11.212-hdd7fb2f_0 
  blas               pkgs/main/osx-arm64::blas-1.0-openblas 
  boost-cpp          pkgs/main/osx-arm64::boost-cpp-1.82.0-h48ca7d4_2 
  botocore           pkgs/main/osx-arm64::botocore-1.36.3-py312hca03da5_0 
  bottleneck         pkgs/main/osx-arm64::bottleneck-1.4.2-py312ha86b861_0 
  cachetools         pkgs/main/osx-arm64::cachetools-5.5.1-py312hca03da5_0 
  frozenlist         pkgs/main/osx-arm64::frozenlist-1.5.0-py312h80987f9_0 
  fsspec             pkgs/main/osx-arm64::fsspec-2024.12.0-py312hca03da5_0 
  gflags             pkgs/main/osx-arm64::gflags-2.2.2-h313beb8_1 
  glog               pkgs/main/osx-arm64::glog-0.5.0-h313beb8_1 
  importlib_resourc~ pkgs/main/osx-arm64::importlib_resources-6.4.0-py312hca03da5_0 
  jmespath           pkgs/main/osx-arm64::jmespath-1.0.1-py312hca03da5_0 
  joblib             pkgs/main/osx-arm64::joblib-1.4.2-py312hca03da5_0 
  libboost           pkgs/main/osx-arm64::libboost-1.82.0-h0bc93f9_2 
  libbrotlicommon    pkgs/main/osx-arm64::libbrotlicommon-1.0.9-h80987f9_9 
  libbrotlidec       pkgs/main/osx-arm64::libbrotlidec-1.0.9-h80987f9_9 
  libbrotlienc       pkgs/main/osx-arm64::libbrotlienc-1.0.9-h80987f9_9 
  libevent           pkgs/main/osx-arm64::libevent-2.1.12-h02f6b3c_1 
  libgfortran        pkgs/main/osx-arm64::libgfortran-5.0.0-11_3_0_hca03da5_28 
  libgfortran5       pkgs/main/osx-arm64::libgfortran5-11.3.0-h009349e_28 
  libgrpc            pkgs/main/osx-arm64::libgrpc-1.62.2-h62f6fdd_0 
  libopenblas        pkgs/main/osx-arm64::libopenblas-0.3.21-h269037a_0 
  libthrift          pkgs/main/osx-arm64::libthrift-0.15.0-h73c2103_2 
  libxgboost         pkgs/main/osx-arm64::libxgboost-2.1.1-h313beb8_0 
  llvm-openmp        pkgs/main/osx-arm64::llvm-openmp-14.0.6-hc6e5704_0 
  multidict          pkgs/main/osx-arm64::multidict-6.1.0-py312h80987f9_0 
  numexpr            pkgs/main/osx-arm64::numexpr-2.10.1-py312h5d9532f_0 
  numpy              pkgs/main/osx-arm64::numpy-1.26.4-py312h7f4fdc5_0 
  numpy-base         pkgs/main/osx-arm64::numpy-base-1.26.4-py312he047099_0 
  orc                pkgs/main/osx-arm64::orc-2.0.1-h937ddfc_0 
  pandas             pkgs/main/osx-arm64::pandas-2.2.3-py312hcf29cfe_0 
  propcache          pkgs/main/osx-arm64::propcache-0.3.1-py312h80987f9_0 
  py-xgboost         pkgs/main/osx-arm64::py-xgboost-2.1.1-py312hca03da5_0 
  pyarrow            pkgs/main/osx-arm64::pyarrow-17.0.0-py312h313beb8_1 
  python-tzdata      pkgs/main/noarch::python-tzdata-2023.3-pyhd3eb1b0_0 
  pytimeparse        pkgs/main/osx-arm64::pytimeparse-1.1.8-py312hca03da5_0 
  re2                pkgs/main/osx-arm64::re2-2022.04.01-hc377ac9_0 
  retrying           pkgs/main/noarch::retrying-1.3.3-pyhd3eb1b0_2 
  s3fs               pkgs/main/osx-arm64::s3fs-2024.12.0-py312hca03da5_0 
  scikit-learn       pkgs/main/osx-arm64::scikit-learn-1.5.2-py312h313beb8_0 
  scipy              pkgs/main/osx-arm64::scipy-1.15.2-py312h14fe6a6_1 
  snappy             pkgs/main/osx-arm64::snappy-1.2.1-h313beb8_0 
  sniffio            pkgs/main/osx-arm64::sniffio-1.3.0-py312hca03da5_0 
  snowflake-ml-pyth~ pkgs/main/osx-arm64::snowflake-ml-python-1.8.1-py312hca03da5_100 
  sqlparse           pkgs/main/osx-arm64::sqlparse-0.5.2-py312hca03da5_0 
  threadpoolctl      pkgs/main/osx-arm64::threadpoolctl-3.5.0-py312h989b03a_0 
  utf8proc           pkgs/main/osx-arm64::utf8proc-2.6.1-h80987f9_1 
  wrapt              pkgs/main/osx-arm64::wrapt-1.17.0-py312h80987f9_0 
  xgboost            pkgs/main/osx-arm64::xgboost-2.1.1-py312hca03da5_0 
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
/Users/dgillis/miniconda3/lib/python3.12/site-packages/conda/base/context.py:201: FutureWarning: Adding 'defaults' to channel list implicitly is deprecated and will be removed in 25.3. 

To remove this warning, please choose a default channel explicitly with conda's regular configuration system, e.g. by adding 'defaults' to the list of channels:

  conda config --add channels defaults

For more information see https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/use-condarc.html

  deprecated.topic(
/Users/dgillis/miniconda3/lib/python3.12/site-packages/conda/base/context.py:201: FutureWarning: Adding 'defaults' to channel list implicitly is deprecated and will be removed in 25.3. 

To remove this warning, please choose a default channel explicitly with conda's regular configuration system, e.g. by adding 'defaults' to the list of channels:

  conda config --add channels defaults

For more information see https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/use-condarc.html

  deprecated.topic(
Channels:
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/dgillis/miniconda3

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
