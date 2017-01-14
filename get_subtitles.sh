#!/bin/bash

filename="$1"
IFS=$'\n'
CNT=0

for line in `cat "$filename"`
do
    CNT=$((CNT+1))
    echo [$CNT] collecting subtitle: "$line"
    timeout 40 subliminal download -l zh -e utf8 -d subtitles "$line" &
    if [ $CNT -eq 20 ]
    then
        wait
        CNT=0
    fi
done

exit 0
