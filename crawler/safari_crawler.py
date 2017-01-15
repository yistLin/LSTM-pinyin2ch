#!/usr/bin/env python
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import subprocess
from pyunpack import Archive
from rarfile import RarFile

srt = ['.big5', '繁体', '.cht', ' Cht']
def check(file):
  if '.srt' not in file:
    return False

  flag = False
  for i in srt:
    if i in file:
      flag = True
  return flag

"""
Crawling the data from subtitle website
"""

search = "http://subhd.com/a/"
start = 321412
end = 358911
driver = webdriver.Safari()

for i in reversed(range(start, end)):
  query = search + str(i)

  html = urlopen(query)
  soup = BeautifulSoup(html.read(), "html.parser")

  flag = 0
  for results in soup.select('.box'):
    ret = results.get_text()
    if (ret.find('繁体') != -1):
      flag = 1
      #print( 'Yaaa' + '\n===\n')

  if flag:
    print(query)
    driver.implicitly_wait(10)
    driver.get(query)
    elem = driver.find_element_by_id("down")
    if elem != None:
      elem.click()
    #print(elem)
try:
  element = WebDriverWait(driver, 10).until(
      EC.presence_of_element_located((By.ID, "myDynamicElement"))
  )
finally:
  driver.quit()


"""
Handle Unrar + Unzip
"""
'''
subdir = '/Users/anonymous_wrytus/Downloads/Subtitles/'
output = subprocess.check_output(['ls', subdir])
ls = output.decode('utf-8').split()
#print(ls)

for item in ls:
  item = subdir + item
  try:
    Archive(item).extractall(subdir)
  except:
    with RarFile(item) as rf:
      rf.extractall(subdir)
  subprocess.check_output(['rm', item])


"""
Pull out the traditional Chinese subtitles
"""

outcome = '/Users/anonymous_wrytus/Downloads/output/'
output = subprocess.check_output(['ls', subdir])
ls = output.decode('utf-8').split()
print(ls)

for item in ls:
  item = subdir + item
  print(item)
  if os.path.isdir(item):
    item = item + '/'
    folder_raw = subprocess.check_output(['ls', item])
    folder = folder_raw.decode('utf-8').split()
    for file in folder:
      if check(file):
        tmp = item + file
        subprocess.check_output(['mv', tmp, outcome])

  else:
    if check(item):
      subprocess.check_output(['mv', item, outcome])
'''