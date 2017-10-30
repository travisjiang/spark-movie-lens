#!/usr/bin/env bash

hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }

file_name="ml-10m.zip"
#file_name="ml-latest.zip"
file_url="http://files.grouplens.org/datasets/movielens/$file_name"
wget $file_url
DESTINATION="./datasets/"
mkdir -p $DESTINATION
mv $file_name $DESTINATION
cd $DESTINATION
unzip -o $file_name
