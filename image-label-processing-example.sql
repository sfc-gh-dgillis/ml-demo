CREATE DATABASE IF NOT EXISTS st_db
    COMMENT = 'Demo database';

CREATE SCHEMA IF NOT EXISTS st_db.st_schema
    COMMENT = 'Demo database';

GRANT USAGE ON DATABASE st_db TO ROLE demo_rw;
GRANT USAGE ON SCHEMA st_db.st_schema TO ROLE demo_rw;

GRANT ROLE demo_data_engineer TO USER cdevo;
GRANT ROLE demo_data_engineer TO ROLE sysadmin;

GRANT ALL ON SCHEMA st_db.st_schema TO ROLE demo_data_engineer;
GRANT CREATE STAGE ON SCHEMA st_db.st_schema TO ROLE demo_data_engineer;
GRANT CREATE NOTEBOOK ON SCHEMA st_db.st_schema TO ROLE demo_data_engineer;
GRANT CREATE SERVICE ON SCHEMA st_db.st_schema TO ROLE demo_data_engineer;
GRANT ALL ON SCHEMA st_db.st_schema TO ROLE demo_data_engineer;

CREATE OR REPLACE STAGE st_db.st_schema.data_stage_ray/images/
    DIRECTORY = (ENABLE = TRUE);

CREATE OR REPLACE STAGE st_db.st_schema.data_stage_ray/labels/
    DIRECTORY = (ENABLE = TRUE);

CREATE COMPUTE POOL IF NOT EXISTS demo_compute_pool
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S
  INITIALLY_SUSPENDED =  TRUE
COMMENT = 'Compute pool for demo database, particularly for ML image processing demo';

GRANT USAGE ON COMPUTE POOL demo_compute_pool TO ROLE demo_rw;
