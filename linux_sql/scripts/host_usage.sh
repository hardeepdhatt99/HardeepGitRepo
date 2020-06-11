#! /bin/bash

# Assign arguments
psql_host=$1
psql_port=$2
db_name=$3
psql_user=$4
psql_password=$5

# Validate arguments
if [ $# -ne 5 ]; then
  echo "Invalid number of arguments"
  exit 1
fi

# Parse hardware usage information using CLI
timestamp=$(date +%Y-%m-%d\ %H:%M:%S)
vms=$(vmstat -w)
memory_free=$(echo "$vms" | tail -n +3 | awk '{print $4}' | xargs)
cpu_idle=$(echo "$vms" | tail -n +3 | awk '{print $15}' | xargs)
cpu_kernel=$(echo "$vms" | tail -n +3 | awk '{print $14}' | xargs)
disk_io=$(echo "$vms" | tail -n +3 | awk '{print $14}' | xargs)
disk_available=$(df -BM / | tail -n +2 | awk '{print $4}' | sed 's/.$//' | xargs)

# Construct INSERT statement
insert_stmt="INSERT INTO host_usage (timestamp,host_id,memory_free,cpu_idle,cpu_kernel,disk_io,disk_available)
VALUES ('${timestamp}', (SELECT id FROM host_info), ${memory_free}, ${cpu_idle}, ${cpu_kernel}, ${disk_io}, ${disk_available});"

# Execute INSERT statement
export PGPASSWORD=$psql_password
psql -h $psql_host -p $psql_port -U $psql_user -d $db_name -c "$insert_stmt"
exit $?