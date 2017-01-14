#!/usr/bin/env python
import os, sys
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

if len(sys.argv) != 3:
  print("Usage: ./{} [start] [end]".format(sys.argv[0]))
  sys.exit(-1)

link = "https://www.rottentomatoes.com/top/bestofrt/?year="
yearStart = int(sys.argv[1])
yearEnd = int(sys.argv[2])

f = open('movie_list.txt', 'a')

for i in reversed(range(yearStart, yearEnd)):
  query = '{}{}'.format(link, i)
  print(query)
  html = urlopen(query)
  soup = BeautifulSoup(html.read(), "html.parser")

  ite = 0
  for results in soup.select('.table td .unstyled.articleLink'):
    ite += 1
    #print("{}>>>{}".format(ite, repr(results.string.strip())))
    movie_name = re.sub(r'(.*?)\(.*\)', r'\1', results.string.strip())
    #print("{}>>>{}".format(ite, repr(movie_name.strip())))
    f.write(movie_name.strip() + '\n')
f.close()
