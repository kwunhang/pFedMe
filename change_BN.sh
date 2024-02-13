#! /bin/bash
if [ $1 != "" ]; then
    BASEDIR=$(dirname $0)
    echo "change Batch Normalization approach to $1";
    cp -f "$BASEDIR/FLAlgorithms/BN/$1/userbase.py" "$BASEDIR/FLAlgorithms/users/userbase.py"
    echo "end of script";
fi