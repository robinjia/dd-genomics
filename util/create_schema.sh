#! /bin/sh
# 
# Create the database tables
# 
source ./env_local.sh
echo $GDD_HOME
SCHEMA_FILE="${GDD_HOME}/util/schema.sql"
if [ ! -r ${SCHEMA_FILE} ]; then
	echo "$0: ERROR: schema file is not readable" >&2
	exit 1
fi

if [ "$1" == "pg" ]; then
	sed 's/DISTRIBUTED BY (doc_id)//g' ${SCHEMA_FILE} | psql -X --set ON_ERROR_STOP=1 -d ${DBNAME} 
else
	psql -X --set ON_ERROR_STOP=1 -d ${DBNAME} -f ${SCHEMA_FILE} 
fi
