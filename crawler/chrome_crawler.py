#!/usr/bin/env python
from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os

search = "http://subhd.com/a/"

start = 48
end = 50
chromedriver = "./dsp_final/chromedriver"
driver = webdriver.Chrome(chromedriver)
os.environ["webdriver.chrome.driver"] = chromedriver

for i in range(start, end):
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
    driver.get(query)
    elem = driver.find_element_by_id("down").click()
    #print(elem)

driver.quit()