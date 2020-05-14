if (!require("tidyverse")) install.packages("tidyverse")
if (!require("rvest"))     install.packages("rvest")
if (!require("stringr"))   install.packages("stringr")
if (!require("tidytext"))  install.packages("tidytext")
if (!require("tictoc"))    install.packages("tictoc")
if (!require("httr"))      install.packages("httr")
if (!require("rjson"))      install.packages("rjson")


library(tidyverse)
library(rvest)
library(stringr)
library(tidytext)
library(tictoc)
library(httr)
library(rjson)

################################
# Scrap artists
################################

# JSON file for popular artists in the last 30 days
json_file <- "https://www.wikiart.org/en/App/Search/popular-artists?json=3&amp;layout=new"

json_data <- fromJSON(file=json_file)

# Read the html page of the 60 popular artists
h <- read_html(json_data$ArtistsHtml)

# Number of artists to scrape
Nartists <- 10

# Contains the number of paintings and the name of the artist
temp_list <- h %>% html_nodes('a') %>% html_attr('title')

# Get the urls of the artists
urls  <- h %>% html_nodes('a') %>% html_attr('href') %>% .[seq(1,length(temp_list),3)]

# Make name-id to be the unique identifier
id <- substr(urls,5,nchar(urls))

# Add the prefix to complete the url
urls <- paste0('https://www.wikiart.org',urls)

# Get the artist name
name = temp_list[seq(1,length(temp_list),3)]

# Get the number of paintings
numberofpaintings <- temp_list[seq(2,length(temp_list),3)] %>%
  gsub("(\\d*).*","\\1",.) %>% as.numeric()

# Create a data frame to write the information collected in a csv file
csv_df <- data.frame(id = id, name = name,
                     numberofpaintings = numberofpaintings,
                     url = urls,stringsAsFactors = FALSE)

# Select the first 10 artists with more than 600 paintings
csv_df  <- csv_df %>% filter(numberofpaintings > 600) %>% .[1:Nartists,]

# Clean up              
rm(temp_list, numberofpaintings, name, urls,id)      

# Create a directory data to store the csv file and the images
if(!dir.exists('data')) dir.create('data')

# Create a sub-directory data to store images
if(!dir.exists('data/rawdata')) dir.create('data/rawdata')

# Write to a csv file
write.csv(csv_df,file = paste0('data/artists','.csv'),row.names = FALSE)

# Get images for each artist
for (i in 1:nrow(csv_df)){
  
  # url list for paintings
  url <- paste0(csv_df$url[i],'/all-works/text-list')
  # read the html
  h <- read_html(url) 
  
  # Node containing url for the paintings
  url_node_data <- h %>% html_nodes('a') %>% html_attr('href') 
  
  # Create a directory with artist name
  if(!dir.exists(paste0('data/rawdata/',csv_df$id[i]))) dir.create(paste0('data/rawdata/',csv_df$id[i]))
  
  # Loop through all the painting pages
  for (j in 1:csv_df$numberofpaintings[i]){
    
    # Url for each painting
    url1 <-  paste0('https://www.wikiart.org',url_node_data[41+j])
    # Read the html page
    h1 <- read_html(url1)
    
    # Image Url
    image_link <- h1 %>% html_nodes('img.ms-zoom-cursor') %>% html_attr('src')
    
    # Get the smaller version of the URL
    temp_link <- unlist(str_split(image_link,"!", n= 2))[1]
    
    # Links have three formats, jpg, jpeg and .png, Store depending on the format
    if(str_detect(temp_link,regex('.Jpg', ignore_case = T))){
      image_link <- paste0(temp_link,'!PinterestSmall.jpg')
    } else if(str_detect(temp_link,regex('.Jpeg', ignore_case = T))){
      image_link <- paste0(temp_link,'!PinterestSmall.jpeg')
    } else(str_detect(temp_link,regex('.png', ignore_case = T))){
      image_link <- paste0(temp_link,'!PinterestSmall.png')
    }
    # Download the url
    if(str_detect(temp_link,regex('.Jpg', ignore_case = T)) | 
       str_detect(temp_link,regex('.Jpeg', ignore_case = T))){
      download.file(image_link,paste0('data/rawdata/',csv_df$id[i],'/',
                                      csv_df$id[i],'_',j,'.jpg'), mode = 'wb', method = 'curl')
    } else{
      download.file(image_link,paste0('data/rawdata/',csv_df$id[i],'/',
                                      csv_df$id[i],'_',j,'.png'), mode = 'wb', method = 'curl')     
    }
    # Wait for .25 sec between loops to reduce error between consecutive requests
    Sys.sleep(.25)
  }
}
