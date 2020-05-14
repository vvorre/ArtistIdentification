if (!require("tidyverse")) install.packages("tidyverse")
if (!require("stringr"))   install.packages("stringr")
if (!require("BiocManager"))    install.packages("BiocManager")
if (!require("EBImage"))   BiocManager::install("EBImage")
if (!require("httr")) install.packages("httr")
if (!require("data.tree"))   install.packages("data.tree")

library(EBImage)
library(tidyverse)
library(stringr)
library(httr)
library(data.tree)
# temp file to store the zip
temp <- tempfile(fileext = ".zip")
# write the content of the download to a binary file
writeBin(content(GET('https://www.dropbox.com/s/5edv6f214lmhn83/data_artists.zip?dl=1'), "raw"), temp)
# unzip it to data
unzip(temp , exdir = 'data' )
# rm the tempfile
rm(temp)

# Read the CSV file containing information about artists and the images
art_info_df <- read.csv2('data/artists.csv',sep=',',stringsAsFactors = FALSE)

# Glimpse at the data
glimpse(art_info_df)

# Name of the artists
art_info_df$name

# Plot of paintings per artist
art_info_df %>% 
  ggplot(aes(reorder(name, numberofpaintings), numberofpaintings, fill= numberofpaintings))+
  geom_bar(stat = "identity") + coord_flip() +
  scale_fill_gradient(low = "yellow", high = "red")+
  ylab("Number of paintings") +
  xlab("Name of the artist") 

# Read some sample images from picasso
par(mfrow = c(3, 2),oma=c(0,0,2,0))
for (i in seq(15,20)){
  plot(readImage(paste0('data/rawdata/pablo-picasso/pablo-picasso_',i,'.jpg')))
}
title("Picasso", outer=TRUE) 
par(mfrow = c(1, 1))

# We observe that there is an uneven distribution of paintings per artist
# We solve this by choosing 400 images per artist. We perform the two operations
# 1. Downsample the images to 400 
# 2. We also crop all the images to 224x224 so that they have the same size
# This removes some information about the artist. Due to space contrainsts, we choose
# the downsample method. Other methods exists such as data augmentation or bootstrapping approaches

# The zip file also contains the validation set, so no need to process the data again
if(!dir.exists('data/procdata')){
# Create a new directory for processed data
dir.create('data/procdata')

# train = 75% - 300 images, validation = 12.5% - 50 images, and test = 12.5% - 50 images
# Create train, validation and test folders for data
if(!dir.exists('data/procdata/train')) dir.create('data/procdata/train')
if(!dir.exists('data/procdata/validation')) dir.create('data/procdata/validation')
if(!dir.exists('data/procdata/test')) dir.create('data/procdata/test')
ntrain <- 300
nvalid <- 50
ntest  <- 50
npix   <- 224 # number of pixels in the final image npix x npix
# set seed
set.seed(1)

for (j in 1:nrow(art_info_df)){
# Create directory for artist in the train,validation and test folder  
if(!dir.exists(paste0('data/procdata/train/',art_info_df$id[j]))) 
  dir.create(paste0('data//procdata/train/',art_info_df$id[j]))
  
if(!dir.exists(paste0('data/procdata/validation/',art_info_df$id[j]))) 
    dir.create(paste0('data//procdata/validation/',art_info_df$id[j])) 

if(!dir.exists(paste0('data/procdata/test/',art_info_df$id[j]))) 
    dir.create(paste0('data//procdata/test/',art_info_df$id[j]))  

# Indices for files 
filter_indx <- seq(1:art_info_df$numberofpaintings[j])
# index for training samples  
train_idx <- sample(filter_indx,ntrain,replace = FALSE)
# index for validation samples over the remaing indices
valid_idx <- sample(setdiff(filter_indx,train_idx), nvalid,replace = FALSE)
# Left over indices
rem_idx <- setdiff(setdiff(filter_indx,train_idx),valid_idx) 
# index for test samples over the remaing indices
test_idx <- sample(rem_idx, ntest,replace = FALSE)

# Write data in train folder
lapply(seq(1,length(train_idx)), function(indx){
  # Read image prefix
  rdimgpre <- paste0('data/rawdata/',art_info_df$id[j],'/',art_info_df$id[j],'_')
  # Read the image
  img <-  readImage(paste0(rdimgpre,train_idx[indx],'.jpg'))
  # Read the dimensions
  imgdim <- dim(img)
  # Min dimensions of the image
  dim_min <- min(imgdim[1],imgdim[2])
  # Crop the image from the center to the minimum dimension
  img_crop <- img[(round((imgdim[1]-dim_min)/2)+1):(round((imgdim[1]+dim_min)/2)),
                  (round((imgdim[2]-dim_min)/2)+1):(round((imgdim[2]+dim_min)/2)),]
  # Resize the image to 224 x 224 dimension
  img_res <- resize(img_crop, w = npix, h = npix)
  # Write image prefix
  wtimgpre <- paste0('data/procdata/train/',art_info_df$id[j],'/',art_info_df$id[j],'_')  
  # write image to train directory
  writeImage(img_res, paste0(wtimgpre,indx ,'.jpg'))  
  })

# Write data in validation folder
lapply(seq(1,length(valid_idx)), function(indx){
  # Read image prefix
  rdimgpre <- paste0('data/rawdata/',art_info_df$id[j],'/',art_info_df$id[j],'_')
  # Read the image
  img <-  readImage(paste0(rdimgpre,valid_idx[indx],'.jpg'))
  # Read the dimensions
  imgdim <- dim(img)
  # Min dimensions of the image
  dim_min <- min(imgdim[1],imgdim[2])
  # Crop the image from the center to the minimum dimension
  img_crop <- img[(round((imgdim[1]-dim_min)/2)+1):(round((imgdim[1]+dim_min)/2)),
                  (round((imgdim[2]-dim_min)/2)+1):(round((imgdim[2]+dim_min)/2)),]  
  # Resize the image to 224 x 224 dimension
  img_res <- resize(img_crop, w = npix, h = npix)
  # Write image prefix
  wtimgpre <- paste0('data/procdata/validation/',art_info_df$id[j],'/',art_info_df$id[j],'_')  
  # write image to validation directory
  writeImage(img_res, paste0(wtimgpre,indx ,'.jpg'))  
})

# Write data in test folder
lapply(seq(1,length(test_idx)), function(indx){
  # Read image prefix
  rdimgpre <- paste0('data/rawdata/',art_info_df$id[j],'/',art_info_df$id[j],'_')
  # Read the image
  img <-  readImage(paste0(rdimgpre,test_idx[indx],'.jpg'))
  # Read the dimensions
  imgdim <- dim(img)
  # Min dimensions of the image
  dim_min <- min(imgdim[1],imgdim[2])
  # Crop the image from the center to the minimum dimension
  img_crop <- img[(round((imgdim[1]-dim_min)/2)+1):(round((imgdim[1]+dim_min)/2)),
                  (round((imgdim[2]-dim_min)/2)+1):(round((imgdim[2]+dim_min)/2)),]
  # Resize the image to 224 x 224 dimension
  img_res <- resize(img_crop, w = npix, h = npix)
  # Write image prefix
  wtimgpre <- paste0('data/procdata/test/',art_info_df$id[j],'/',art_info_df$id[j],'_')  
  # write image to test directory
  writeImage(img_res, paste0(wtimgpre,indx ,'.jpg'))  
})

}

}
# Read sample images after cropping and resizing
par(mfrow = c(3, 2),oma=c(0,0,2,0))
for (i in seq(1,6)){
  plot(readImage(paste0('data/procdata/train/pablo-picasso/pablo-picasso_',i,'.jpg')))
}
title("Picasso", outer=TRUE) 
par(mfrow = c(1, 1))

par(mfrow = c(3, 2),oma=c(0,0,2,0))
for (i in seq(1,6)){
  plot(readImage(paste0('data/procdata/train/claude-monet/claude-monet_',i,'.jpg')))
}
title("Monet", outer=TRUE) 
par(mfrow = c(1, 1))

# The file structure is given as below
path <- c(
  "data/procdata/train/artist/images", 
  "data/procdata/validation/artist/images", 
  "data/procdata/test/artist/images"
)
(mytree <- data.tree::as.Node(data.frame
                              (pathString = path)))


