# Contribution Of Characters To The MCU
This project calculates the contribution of each character to the Marvel Cinematic Universe in terms of how much time they spend on screen. The TF-IDF ranking algorithm is used here for the purpose.

### Requirements
- Python version: 3.7.4
- Python modules: bs4, requests, collections, pandas, numpy, re

### Idea
- The main idea here was to find out how much screen time each character has faced in relation to the entire [Marvel Cinematic Universe](https://en.wikipedia.org/wiki/Marvel_Cinematic_Universe). The movies covered here are only those that are part of [the Infinity Saga](https://en.wikipedia.org/wiki/List_of_Marvel_Cinematic_Universe_films#The_Infinity_Saga).
- The [TF-IDF algorithm](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is used to rank the characters based on their screen time.
- In the traditional sense, TF-IDF is used to rank the relevance of documents based on the term frequencies. Here, I have altered the algorithm to incorporate screen time of characters
- TF = Time spent by character in a movie / Total running length of the movie
- IDF = Total number of movies considered / natural_log(Number of movies the character appears in)
- A result sheet is generated, where TF-IDF scores for each character has been calculated considering his/her movies only, and the entire MCU

### Steps
- Just run ```mcu.py``` to generate the results.
- ```characters.csv``` file needs to be placed in the same directory
- Three CSV files - ```character_screen_time_every_movie.csv```: Matrix with each character's screen time for each movie, ```character_tf_idf_every_movie.csv```: TF-IDF scores for each character over each movie, and ```character_tf_idf_mcu_contribution.csv```: Mean TF-IDF scores calculated for each character over all his/her movies, and over the entire MCU - are generated in the same directory.

### Further work to be done
- Visualisation of the data in some graphical format using matplotlib or bokeh.
- Understanding the results clearly (I'm really interested in what these values really mean)
- Maybe a written analysis of the results as a blog post (hopefully)

Data for the analysis obtained from [IMDB](https://www.imdb.com/list/ls066620113/?sort=list_order,asc&st_dt=&mode=detail&page=1&ref_=ttls_vm_dtl) created by [ninewheels0](https://www.imdb.com/user/ur51880615/?ref_=_usr)
