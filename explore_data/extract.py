import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import collections
import pandas as pd

# get articles
tree = ET.parse('articles-training-20180831.xml')
root = tree.getroot()
# use Beautiful soup package to extract XML contents

# store article to a plain text file
f = open("800k_dataset.txt", 'w')
for child in root:
    title = child.attrib["title"]
    xmlstr = ET.tostring(child, encoding='utf8', method='xml')
    soup = BeautifulSoup(xmlstr)    # txt is simply the a string with your XML file
    pageText = soup.findAll(text=True)
    one_string =' '.join(pageText).replace("?xml version='1.0' encoding='utf8'?","")
    rm_string = re.sub(r'\n', '', one_string)
    rm_string = re.sub(r'\t', ' ', rm_string)
    f.write(child.attrib["id"]+"\t"+title + " " + rm_string+"\n")
f.close()

# get articles
tree = ET.parse('articles-validation-20180831.xml')
root = tree.getroot()
# use Beautiful soup package to extract XML contents

# store article to a plain text file
f = open("200k_validation.txt", 'w')
for child in root:
    title = child.attrib["title"]
    xmlstr = ET.tostring(child, encoding='utf8', method='xml')
    soup = BeautifulSoup(xmlstr)    # txt is simply the a string with your XML file
    pageText = soup.findAll(text=True)
    one_string =' '.join(pageText).replace("?xml version='1.0' encoding='utf8'?","")
    rm_string = re.sub(r'\n', '', one_string)
    rm_string = re.sub(r'\t', ' ', rm_string)
    f.write(child.attrib["id"]+"\t"+title + " " + rm_string+"\n")
f.close()