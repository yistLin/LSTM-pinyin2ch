#!/usr/bin/env python
import os, sys
import requests
from bs4 import BeautifulSoup as bf
import re

f = open('_thenewslens_list.txt', 'a')

link = "https://www.thenewslens.com/article/"
start = 1
end = 60000

r = requests.session()

for i in range(start, end):
  query = '{}{}'.format(link, i)
  print(query)
  try:
    result = r.get(query)
  except:
    print('error')
    continue
  soup = bf(result.text, 'html.parser')
  #print(soup)

  ite = 0
  for results in soup.select('.article-content > p'):
    ite += 1
    try:
      #print("{}>>>{}".format(ite, repr(results.string.strip())))
      text = results.string.strip()
    except:
      continue

    remove_symbol = re.split(r'[^\u4e00-\u9fff|^0-9)]', text)
    for sen in remove_symbol:
      print(sen)
      if len(sen) != 0:
        f.write(sen.strip() + '\n')

f.close()