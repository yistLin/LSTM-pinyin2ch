#!/usr/bin/env python
import os, sys
import re

if len(sys.argv) != 3:
  print("Usage: ./remove_garbage.py [garbage filename] [output filename]")
  sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]
f = open(infile, 'r')
o = open(outfile, 'w')

line_num = 1
wrong_thing = ""
for line in f.readlines():
  flag = 0  # flag: Check if this sentence should be abandoned
  char_list = line.split()
  for char in char_list:
    #print(i, i.encode('raw_unicode_escape'))
    match = re.search(r'[^\u4e00-\u9fff|^0-9)]', char)
    if match:
      flag = 1
      wrong_thing = line
      break

  if not flag:
    o.write(line)
  else:
    print(line_num, wrong_thing)
  line_num += 1

f.close()
o.close()