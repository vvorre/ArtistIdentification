# Prerequisites
Anaconda software ((https://www.anaconda.com/)) should be installed prior to running the code of the neural network. 

# Files
The files are listed in the order to run.

scrape_artists.R - code to scrape paintings from WikiArt: This is optional to run since preprocess.R automatically downloads the dataset from Dropbox.

preprocess.R - code to preprocess the downloaded images

train_cnn.R  - training the CNN and extracting the features

analysis.R   - Training other machine learning models and results

# Reports
Artist_Report_pdf - pdf report of the project

Artist_Report_html - HTML report of the project

HTML rendered output is at available [here]https://rawcdn.githack.com/vvorre/ArtistIdentification/f5b9d0ba949e099a45dbfd9255851e58f6eb22d2/Artists_Report_html.html)

# Description
The goal of the project is to predict artist from a painting. The dataset consists of 4000 paintings from 10 popular artists where each artist has 400 paintings. We use a convolution neural network (Resnet18) to build a model as well as extract features from the images. The extracted features are also trained on other machine learning models to improve performance. 
