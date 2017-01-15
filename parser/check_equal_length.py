#!/usr/bin/env python
import os, sys
import re

if len(sys.argv) != 3:
  print("Usage: ./check_equal_length.py [word&pinyin filename] [output filename]")
  sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]
f = open(infile, 'r')
o = open(outfile, 'w')

for line in f.readlines():
  tmp = line.split('\t')
  chinese = tmp[0]
  pinyin = tmp[1]
  #print(chinese.split())
  #print(pinyin.split() , '\n===')
  if len(chinese.split()) == len(pinyin.split()):
    o.write(line)

f.close()
o.close()