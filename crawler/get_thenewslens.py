#!/usr/bin/env python
import os, sys
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import re

f = open('thenewslens_list.txt', 'a')


"""
Get articles from thenewslens
"""

link = "https://www.thenewslens.com/article/"
start = 1
end = 60000

for i in range(start, end):
  query = '{}{}'.format(link, i)
  print(query)
  req = Request(query, headers={'User-Agent': 'Mozilla/5.0'})
  try:
    html = urlopen(req)
  except:
    continue
  soup = BeautifulSoup(html.read(), "html.parser")


  ite = 0
  for results in soup.select('.article-content > p'):
    ite += 1
    #print(repr(results.string))
    try:
      #print("{}>>>{}".format(ite, repr(results.string.strip())))
      text = results.string.strip()
    except:
      continue

    remove_symbol = re.split(r'[^\u4e00-\u9fff|^0-9)]', text)
    for sen in remove_symbol:
      if len(sen) != 0:
        f.write(sen.strip() + '\n')
    #movie_name = re.sub(r'(.*?)\(.*\)', r'\1', results.string.strip())
    #print("{}>>>{}".format(ite, repr(movie_name.strip())))
    #f.write(movie_name.strip() + '\n')

f.close()
