#source: https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57

from urllib.request import urlopen,Request
import requests
import os
import io
import json
from bs4 import BeautifulSoup

DIR = '/Users/subra/Desktop/git/Image_Fun/Pictures'
QUERY = 'parrot'

def get_soup(url,header):
    return BeautifulSoup(urlopen(Request(url,headers=header)),'html.parser')

#query = raw_input("Search Term: ")
query = QUERY
query = query.replace(' ','')
#url = 'https://www.google.com/search?q='+query+'&rlz=1C1CHBF_enUS771US771&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiJ4fmQ6PzcAhVIPN8KHVsoAfsQ_AUICigB&biw=1280&bih=539'
url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
soup = get_soup(url,header)

Images = []
for item in soup.find_all("div",{"class":"rg_meta"}):
    link, Type=json.loads(item.text)["ou"],json.loads(item.text)["ity"]
    Images.append((link,Type))

print(len(Images))

for i,(img,Type) in enumerate(Images[0:len(Images)]):
    req = Request(img, headers = header)
    raw_img = urlopen(req).read()

    if len(Type)==0:
        f = open(os.path.join(DIR,"img"+"_"+str(i)+".jpg"),'wb')
    else:
        f = open(os.path.join(DIR,"img"+"_"+str(i)+"."+Type),'wb')
    f.write(raw_img)
    f.close()
