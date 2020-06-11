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

# Parse all hardware information from bash CLI
hostname=$(hostname -f)
lscpu_out=$(lscpu)
cpu_number=$(echo "$lscpu_out"  | egrep "^CPU\(s\):" | awk '{print $2}' | xargs)
cpu_architecture=$(echo "$lscpu_out" | grep "^Architecture:" | awk '{print $2}' | xargs)
cpu_model=$(echo "$lscpu_out" | grep "^Model\ name:" | awk '{$1 = "";$2 = ""; print $0;}' | xargs)
cpu_mhz=$(echo "$lscpu_out" | grep "^CPU MHz:" | awk '{print $3}' | xargs)
l2_cache=$(echo "$lscpu_out" | grep "L2 cache:" | awk '{print $3}' | sed 's/.$//' | xargs)
total_mem=$(cat /proc/meminfo | grep "^MemTotal:" | awk '{print $2}' | xargs)
timestamp=$(date +%Y-%m-%d\ %H:%M:%S)

# Construct INSERT statements
insert_stmt="INSERT INTO host_info (hostname, cpu_number, cpu_architecture, cpu_model, cpu_mhz, L2_cache, total_mem, "timestamp")
VALUES ('${hostname}',${cpu_number},'${cpu_architecture}', '${cpu_model}', ${cpu_mhz}, ${l2_cache}, ${total_mem}, '${timestamp}');"

# Execute INSERT statement
export PGPASSWORD=$psql_password
psql -h $psql_host -p $psql_port -U $psql_user -d $db_name -c "$insert_stmt"
exit $?