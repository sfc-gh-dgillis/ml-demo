USE ROLE useradmin;

-- create access role
CREATE ROLE IF NOT EXISTS demo_rw
    COMMENT = 'Access role for the demo database with Read and Write permissions to all objects.';

-- create functional role
CREATE ROLE IF NOT EXISTS demo_data_engineer
    COMMENT = 'Functional role for the demo - business function alignment is generally for Data Engineers';

USE ROLE SYSADMIN;

GRANT ROLE demo_data_engineer TO ROLE sysadmin;

CREATE DATABASE IF NOT EXISTS st_db
    COMMENT = 'Demo database';

CREATE SCHEMA IF NOT EXISTS st_db.st_schema
    COMMENT = 'Demo database';

-- create directory stages
CREATE OR REPLACE STAGE st_db.st_schema.data_stage_ray/images/
    DIRECTORY = (ENABLE = TRUE);

CREATE OR REPLACE STAGE st_db.st_schema.data_stage_ray/labels/
    DIRECTORY = (ENABLE = TRUE);

-- create compute  pool
CREATE COMPUTE POOL IF NOT EXISTS demo_compute_pool
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S
  INITIALLY_SUSPENDED =  TRUE
       -- add auto suspend...
COMMENT = 'Compute pool for demo database, particularly for ML image processing demo';

-- TODO
-- schedule training on compute pool that is not associated
-- with the notebook you're running on

-- grants to the demo_rw access role
GRANT USAGE ON DATABASE st_db TO ROLE demo_rw;
GRANT USAGE ON SCHEMA st_db.st_schema TO ROLE demo_rw;
GRANT CREATE STAGE ON SCHEMA st_db.st_schema TO ROLE demo_rw;
GRANT CREATE NOTEBOOK ON SCHEMA st_db.st_schema TO ROLE demo_rw;
GRANT CREATE SERVICE ON SCHEMA st_db.st_schema TO ROLE demo_rw;
GRANT USAGE ON COMPUTE POOL demo_compute_pool TO ROLE demo_rw;
GRANT ALL ON SCHEMA st_db.st_schema TO ROLE demo_rw;
