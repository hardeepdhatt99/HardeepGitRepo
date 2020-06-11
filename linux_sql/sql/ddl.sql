-- connect to host_agent database
\c host_agent;

-- Create table to store host info data
CREATE TABLE IF NOT EXISTS PUBLIC.host_info
    (
        -- Declare table columns and data types
        id               SERIAL PRIMARY KEY,
        hostname         VARCHAR NOT NULL UNIQUE,
        cpu_number       INT NOT NULL,
        cpu_architecture VARCHAR NOT NULL,
        cpu_model        VARCHAR NOT NULL,
        cpu_mhz          NUMERIC NOT NULL,
        L2_cache         INT NOT NULL,
        total_mem        INT NOT NULL,
        "timestamp"      TIMESTAMP NOT NULL
    );

-- Create table to store host usage data
CREATE TABLE IF NOT EXISTS PUBLIC.host_usage
    (
        -- Declare table columns and data types
        "timestamp"      TIMESTAMP NOT NULL,
        host_id          SERIAL NOT NULL,
        memory_free      INT NOT NULL,
        cpu_idle         INT NOT NULL,
        cpu_kernel       INT NOT NULL,
        disk_io          INT NOT NULL,
        disk_available   INT NOT NULL,
        -- Add foreign key constraint
        FOREIGN KEY (host_id) REFERENCES host_info (id)
    );
