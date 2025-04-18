import os
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session

session = Session.builder.configs(SnowflakeLoginOptions(connection_name="demo_dgillis_keypair_auth", login_file="/Users/dgillis/.snowflake/config.toml")).create()


def upload_to_snowflake(stage, file_path, file_type):
    """
    Upload a file to the specified Snowflake stage.

    Args:
        stage (str): Snowflake stage path (e.g., @data_stage/images/train).
        file_path (str): Path to the file to upload.
        file_type (str): Type of file being uploaded
    """
    try:

        session.use_database("ST_DB")
        session.use_schema("ST_SCHEMA")
        session.file.put(f"file://{file_path}", stage, auto_compress=False)
        # session.sql(f"PUT 'file://{file_path}' {stage}")
        print(f"Uploaded file: {file_path} to {stage}")
    except Exception as e:
        print(f"Failed to upload  file: {file_path}. Error: {str(e)}")


def process_and_upload_files(base_dir, image_stage, label_stage):
    """
    Process the directory structure and upload _test.jpg files and their corresponding .txt files to Snowflake.

    Args:
        base_dir (str): Path to the base directory containing PCB data.
        image_stage (str): Snowflake stage path for images.
        label_stage (str): Snowflake stage path for labels.
    """
    for group_folder in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group_folder)
        if not os.path.isdir(group_path):
            continue

        for sub_folder in os.listdir(group_path):
            sub_folder_path = os.path.join(group_path, sub_folder)

            if not os.path.isdir(sub_folder_path):
                continue

            if sub_folder.endswith("_not"):
                continue

            folder_not = os.path.join(group_path, sub_folder + "_not")

            if not os.path.exists(folder_not):
                continue

            for file_name in os.listdir(sub_folder_path):
                if file_name.endswith("_test.jpg"):

                    # Full path of the .jpg file
                    jpg_file_path = os.path.join(sub_folder_path, file_name)

                    # Corresponding .txt file path
                    txt_file_name = os.path.splitext(file_name.replace("_test", ""))[0] + ".txt"
                    txt_file_path = os.path.join(folder_not, txt_file_name)
                    print(txt_file_path)
                    if os.path.exists(txt_file_path):
                        # Upload the .jpg file to the images stage
                        upload_to_snowflake(image_stage_path, jpg_file_path, "image")

                        # Upload the .txt file to the labels stage
                        upload_to_snowflake(label_stage_path, txt_file_path, "label")


# Define local directory where the PCB dataset was downloaded
base_directory = "/Users/dgillis/Documents/dev/github/Surface-Defect-Detection/DeepPCB/PCBData"

# Define Snowflake stage paths
image_stage_path = "@data_stage_ray/images/"
label_stage_path = "@data_stage_ray/labels/"

# Process and upload files
process_and_upload_files(base_directory, image_stage_path, label_stage_path)
