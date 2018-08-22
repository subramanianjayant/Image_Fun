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
```Convnet.py``` contains methods 
```python
Convnet.resize_image
Convnet.process_images
Convnet.get_convnet
```
that can be accessed from other scripts. However, you can also run ```Convnet.py``` directly from the terminal to process the images in ```Image_Fun/Pictures``` and train the Convnet.
