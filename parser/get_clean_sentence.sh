#!/usr/bin/env bash

if [ $# -ne 4 ];then
	echo "Usage: ./get_clean_sentence.sh [input directory] [output directory] [output directory/output filename] [no garbled filename]"
	exit -1
fi

inputdir=$1
outputdir=$2
outfile=$3
no_garbled=$4
IFS=$'\n'

if [ ! -d $outputdir ];then
	mkdir $outputdir
fi

echo "parsing subtitles to a big output sentences file"
./parse_subtitles.sh $inputdir $outputdir $outfile

echo "remove the garbled in the sentences file"
./remove_garbage.py $outfile $nogarbled
