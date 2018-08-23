# Image_Fun
cool stuff with CNNs and GANs

### Scraping the datasets
```ImageScraper.py``` scrapes images off Google Images based on any query you enter. You can change the query in the source code and then run ```ImageScraper.py```:
```bash
Jayant@Jayant-Spectre-x360:/mnt/c/Users/subra/Desktop/git/Image_Fun$ vi ImageScraper.py

QUERY = 'parrot'

Jayant@Jayant-Spectre-x360:/mnt/c/Users/subra/Desktop/git/Image_Fun$ python ImageScraper.py
https://www.google.co.in/search?q=parrot&source=lnms&tbm=isch //search URL
100 //number of images scraped
```
### Processing Images and using Convolutional Neural Network
Simply run ```ConvNet.py``` and a CNN will be trained and tested on the image data provided
in ```Image_Fun/Pictures```. Currently classifies images with ~87% accuracy.

