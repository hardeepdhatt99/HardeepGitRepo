#! /bin/bash

# arguments
action=$1
db_name=$2
db_password=$3
container_name='jrvs-sql'

# validate arguments
if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
    echo "Illegal number of arguments"
    exit 1
fi

# switch to root user and start docker if it is not running
su centos
systemctl status docker || systemctl start docker
docker pull postgres

# Create docker container with user provided name and password
if [ "$action" == "create" ]; then
    # validate arguments
    if [ "$#" -ne 3 ]; then
        echo "Illegal number of arguments, must provide name and password"
        exit 1
    fi

    # Make sure that container does not already exist
    if [ `docker container ls -a -f name=$container_name | wc -l` -eq 2 ]; then
      echo "container already exists"
      exit 1
    fi

    # create a new volume and container using psql image
    docker run --name $container_name -e POSTGRES_PASSWORD=${db_password} -e POSTGRES_USER=${db_name} -d -v pgdata:/var/lib/postgresql/data -p 5432:5432 postgres
    exit $?
fi

# Make sure docker container exists before attempting to start or stop
if [ `docker container ls -a -f name=$container_name | wc -l` -ne 2 ]; then
      echo "No container of that name exists"
      exit 1
fi

# start existing container
if [ "$action" == "start" ]; then
    docker container start $container_name
    exit $?
fi

# stop existing container
if [ "$action" == "stop" ]; then
    docker container stop $container_name
    exit $?
fi

# If the first argument provided is invalid, print error message
echo "Invalid action, first argument must be start, stop, or create"
exit 1

