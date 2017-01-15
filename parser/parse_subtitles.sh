#!/bin/bash

if [ $# -ne 3 ]; then
	echo "Usage: ./parse_subtitles.sh [input directory] [output directory] [output directory/output file name]"
	exit -1
fi

inputdir=$1
outputdir=$2
outputfile=$3
IFS=$'\n'

echo input directory is "$inputdir"
echo output directory is "$outputdir"
echo output file is "$outputfile"

for file in `ls $inputdir`; do
    ./parse_subtitle.py "$inputdir/"$file"" 1>> "$outputfile"
done

cd cover_zh
python2 cover_zh.py ../"$outputdir" cn utf8
