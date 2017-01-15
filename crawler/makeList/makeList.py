#!/usr/bin/env python
import os
import sys
import re

if len(sys.argv) != 2:
  print("Usage: ./makeList.py [filename]\n")
  sys.exit(1)

input = sys.argv[1]
f = open(input, 'r', encoding='latin1')

for line in f.readlines():
  #print(repr(line))
  match = re.search(r'(^.*)\((\d+)\)', line)
  if match:
    rawName = match.group(1)
    year = match.group(2)
    #print(repr(rawName))
    #print(repr(year) + "\n===")
    
  if int(year) < 1980:
    continue

  name = rawName.strip().replace(' ', '.')
  print(repr(name))
  print(repr(year) + "\n===")
  
