# ml-demo

The first and typically most arduous step of Machine Learning is pre-processing. This demo demonstrates how to encode an image with associated image labels as part of pre-processing in ML.

The dataset is a collection of images and their associated labels retrieved from https://github.com/Charmve/Surface-Defect-Detection in the `DeepPCB/PCBData` directory. You should download this repo locally to run this demo. The dataset contains images of printed circuit boards (PCBs) with various defects, such as scratches, dents, and other surface imperfections. The goal is to train a machine learning model to detect these defects in new images.

Take for example the following images from the `DeepPCB/PCBData/group00041/00041` directory:

## A valid image of a PCB with no defects

![Correct Image](https://github.com/Charmve/Surface-Defect-Detection/blob/master/DeepPCB/PCBData/group00041/00041/00041000_temp.jpg)

------

## A defective image of the same PCB with defects:

![Defective Image](https://github.com/Charmve/Surface-Defect-Detection/blob/master/DeepPCB/PCBData/group00041/00041/00041000_test.jpg)


## Defect Classification

The corresponding text file found [here](https://github.com/Charmve/Surface-Defect-Detection/blob/master/DeepPCB/PCBData/group00041/00041_not/00041000.txt) contains the coordinates of the defects in the image (aka labels). The labels are in the form of bounding boxes, which are used to draw boxes around the objects in the image. The labels are in the format `x_min y_min x_max y_max label`, where `x_min` and `y_min` are the coordinates of the top-left corner of the bounding box, `x_max` and `y_max` are the coordinates of the bottom-right corner of the bounding box, and `label` is the class label for the object - in the case of this demo, this is a defect classification dataset.

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

> First four coordinates are `x min` `y min` `x max` `y max` and last number is the type of defect (label). For example, disconnect defect, dot defect, etc.

The following steps will be performed to setup database objects and load the images and their associated labels into a Snowflake stage:

### Step 1 Prerequisites

#### Download the Dataset by cloning the `Charmve/Surface-Defect-Detection` repo

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

The directory you need to reference with image and text files should be similar to the following (note the `PCBData` directory):

```shell
(base) ~/Documents/dev/github/Surface-Defect-Detection/DeepPCB/PCBData git:[master]
ls
group00041      group12100      group13000      group44000      group77000      group92000      trainval.txt
group12000      group12300      group20085      group50600      group90100      test.txt
```

### Step 2 Create Database Objects

Run the `image-label-processing-example-db-objects.sql` script found in the repo root to create the Database, Schema, Stages, Compute Pool, roles and grants required to run the demo. The script will create the following objects:

- An access role named `demo_rw`
- A functional role named `demo_data_engineer`
- A database named `st_db`
- A schema named `st_db.st_schema`
- A stage named `st_db.st_schema.data_stage_ray/images/`
- A stage named `st_db.st_schema.data_stage_ray/labels/`
- A compute pool named `demo_compute_pool`
- Grants required for the above objects

### Step 3 Install the required Python packages

The demo requires the following Python packages to be installed:

- `snowflake-ml-python`
- `snowflake-snowpark-python`

Using the `conda` package manager, you can install the required packages using the following commands:

#### snowflake-ml-python

```shell
conda install snowflake-ml-python
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/ddemo/miniconda3

  added / updated specs:
    - snowflake-ml-python

...

The following packages will be DOWNGRADED:

  cloudpickle                         3.0.0-py312hca03da5_0 --> 2.2.1-py312hca03da5_0 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                                                                                                    
Preparing transaction: done                                                                                                                                         
Verifying transaction: done                                                                                                                                         
Executing transaction: done                                                                                                                                         
```
#### snowflake-snowpark-python

```shell
conda install snowflake-snowpark-python
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/ddemo/miniconda3

  added / updated specs:
    - snowflake-snowpark-python

...

The following packages will be UPDATED:

  snowflake-snowpar~               1.29.1-py312hca03da5_100 --> 1.30.0-py312hca03da5_100 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                                                                                                    
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

### Step 4 Load Image and Label files to Snowflake Stage

In order to load the images and their associated labels into a Snowflake stage where they can be processed, we have provided a Python script `load_files.py` which  will read the images and their associated labels from the local `DeepPCB/PCBData` directory where you put `Charmve/Surface-Defect-Detection` from [Step 1](#step-1-prerequisites) and uploads them to designated Snowflake stages for further processing.

Key Steps:

**Snowflake Session Setup:** Establishes a connection to Snowflake using credentials from a configuration file (setup via Snowflake CLI or SnowSQL).

**File Upload Function:** Defines a function to upload files to specific Snowflake stages (`@data_stage_ray/images/` for images and `@data_stage_ray/labels/` for labels).

**Directory Traversal:** Iterates through the directory structure to locate image files (_test.jpg) and their corresponding label files in a _not subfolder.

**File Upload:** Uploads the identified image and label files to the appropriate Snowflake stages.

#### Update `load_files.py` with your Snowflake credentials

Configure the session and base directory appropriately for your local environment, e.g.: 

```python
session = Session.builder.configs(SnowflakeLoginOptions(connection_name="demo_dgillis_keypair_auth", login_file="/Users/dgillis/.snowflake/config.toml")).create()
...
base_directory = "/Users/dgillis/Documents/dev/github/Surface-Defect-Detection/DeepPCB/PCBData"
```

Run the script:

```shell
python ./load_files.py
```

### Step 5 Process Images and Labels

Run steps in the Notebook on Snowsight. This notebook extracts image to a snowflake table as well as the data from the text file. Creates table maps the image to the text file.

Allows for batch processing of images and text files.

### Step 6 Train Model

TODO - see below

- After we read images, combine with a pytorch pre-trained model 
- Image processing + image training 
