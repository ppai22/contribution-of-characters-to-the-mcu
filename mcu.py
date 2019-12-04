from bs4 import BeautifulSoup
import requests
from collections import OrderedDict
import pandas as pd
import numpy as np
import re


def fetch_data(url):
	"""Fetching data for each movie as in the IMDB link provided"""
	# Fetch HTML content from web page
	page_source = requests.get(url)
	# Provide HTML content to Beautiful Soup
	soup = BeautifulSoup(page_source.content, 'lxml')
	# Initialization of lists for movies, movie characters and movie lengths
	movies = []
	movie_characters = []
	movie_lengths = OrderedDict()
	# Parsing through each movie
	for movie in soup.find_all('div', class_='lister-item mode-detail'):
		# Fetching title of the movie
		movie_title = movie.find('div', class_='lister-item-content').h3.a.text
		# Fetching length of the movie
		movie_length = movie.find('div', class_='lister-item-content').find('span', class_='runtime').text
		movies.append(movie_title)
		movie_lengths[movie_title] = movie_length.split(' ')[0]
		# Fetching characters in the movie
		characters = movie.find('div', class_='list-description').p.text
		movie_characters.append(characters.split('\n'))
	# Creation of an Ordered Dictionary to store the data
	data_dict = OrderedDict()
	# Inputing data into the dictionary
	for i in range(len(movies)):
		data_dict[movies[i]] = movie_characters[i]
	return movie_lengths, data_dict


def clean_data(dict):
	"""Method that returns a dictionary of the format:
		dict = {'Movie1':{'character1':time1, 'character2':time2, ...},
				'Movie2':{..., ...},
				...}"""
	# Initialization of the index dict
	character_movie_time = {}
	# Parsing through data scraped for each movie
	for movie in dict.keys():
		# Parsing through each character in in the movie
		for item in dict[movie]:
			character = item.split('<')[0].strip()
			time = item.split('<')[1].split('>')[0].strip()
			# Adding time for each character to for each movie to the index dict
			if movie in character_movie_time.keys():
				character_movie_time[movie][character] = time
			else:
				character_movie_time[movie] = {character: time}
	# Returning the index dict
	return character_movie_time


def combine_rows(matrix):
	"""Method that combines the duplicate rows that arose due to replacing character IDs"""
	# List of characters
	characters = list(set([character for character in matrix.index]))
	# Iterating over every character
	for character in characters:
		# Creating a mini data frame for each character
		df = matrix.loc[character]
		# If the character row is present just once, iterating over every column would be iterating over every movie
		if len(list(df.index)) != 23:
			# Iterating over every movie for each character
			for movie in matrix.columns:
				# Initializing update value to NaN
				value = np.nan
				li = []
				# Creating a list of values for each movie
				for i in range(len(df.index)):
					li.append(str(df.iloc[i][movie]))
				# Updating the right value
				for item in li:
					if str(item) != 'nan':
						value = item
				# Updating value to the location
				matrix.loc[character, movie] = value
	# If there are n duplicates, n rows would have been created with same values; deleting the duplicates
	matrix = matrix.drop_duplicates()
	return matrix


def remove_characters(data):
	"""Method that returns a pandas dataframe with screen time for characters in each movie. Unimportant chanracters are removed"""
	# Reading data containing list of important characters only and their IDs
	char_id_index = pd.read_csv('.\\characters.csv')
	# Creating aa dict containing character IDs for each character
	names = {old:new for (old, new) in zip(char_id_index['Character Name'], char_id_index['Character ID'])}
	# Converting the data dict into a pandas data frame for easy handling
	char_movie_matrix = pd.DataFrame(data)
	# Renaming characters with their character IDs
	char_movie_matrix.rename(index=names, inplace=True)
	# Creating a list of unimportant characters to be removed
	pop_list = [name for name in char_movie_matrix.index if name not in list(char_id_index['Character ID'])]
	# Removing all unimportant characters from the data frame
	char_movie_matrix = char_movie_matrix.drop(pop_list)
	# Combine rows with same index
	char_movie_matrix = combine_rows(char_movie_matrix)
	# Considering movies in the Infinity Saga only
	del char_movie_matrix['Spider-Man: Far from Home']
	# Returning the character-movie-run_time matrix in the form of a pandas dataframe
	return char_movie_matrix


def calculate_idf(matrix):
	"""Method that calculates the inverse document frequency for each character"""
	# Initialization of document frequency and idf dicts
	doc_freq = dict()
	idf = dict()
	# List of all characters
	characters = [character for character in matrix.index]
	# List of all the movies
	movies = [movie for movie in matrix.columns]
	# N = Total number of movies
	N = len(movies)
	# Iterating over every character
	for character in characters:
		# Finding the document frequency for each character
		if character in doc_freq.keys():
			doc_freq[character] += len([value for value in matrix.loc[character] if str(value).lower() != 'nan'])
		else:
			doc_freq[character] = len([value for value in matrix.loc[character] if str(value).lower() != 'nan'])
	# Finding the IDF value for each character and storing in a dict
	for character in doc_freq.keys():
		idf[character] = np.log(N/doc_freq[character])
	return idf


def convert_time_to_mins(matrix):
	"""Method that converts time of the format mm:ss or :ss or mm to minutes in float"""
	# Iterating over each character
	for character in matrix.index:
		# Iterating over each movie
		for movie in matrix.columns:
			# Current value of the cell in the matrix
			value = matrix.loc[character][movie]
			# Iterating over non NaN values only
			if str(value) != 'nan':
				# To convert strings of the format mm:ss
				if re.match(r'\d+:\d+', value):
					matrix.loc[character][movie] = float(value.split(':')[0]) + float(value.split(':')[1])/60
				# To convert strings of the format :ss
				elif re.match(r'^:\d+', value):
					matrix.loc[character][movie] = float(value.split(':')[1])/60
				# To convert strings of the format mm
				elif re.match(r'^\d+[^:]', value):
					matrix.loc[character][movie] = float(value)
				# Other formats assigned as NaN
				else:
					matrix.loc[character][movie] = np.nan
			# NaN values reassigned as NaN
			else:
				matrix.loc[character][movie] = np.nan
	return matrix


def calculate_tf(matrix, movie_lengths):
	"""Method that calculates TF values for each character over each movie"""
	# Copy the input char_movie_float_time matrix, as output TF matrix would be in the same format
	tf = matrix.copy()
	for character in tf.index:
		for movie in tf.columns:
			# Filling out the cells with TF values for those cells having time values only
			if str(tf.loc[character][movie]) != 'nan':
				tf.loc[character][movie] = tf.loc[character][movie] / float(movie_lengths[movie])
	return tf


def calculate_tf_idf(tf, idf):
	"""Method that calculates the TF-IDF values for each character over every movie"""
	tf_idf = tf.copy()
	for character in tf_idf.index:
		for movie in tf_idf.columns:
			if str(tf_idf.loc[character][movie]) != 'nan':
				tf_idf.loc[character][movie] = tf_idf.loc[character][movie] * idf[character]
	return tf_idf


def calculate_mean_tf_idf(tf_idf):
	"""Method that calculates the mean TF-IDF values over the entire Marvel Cinematic Universe for all characters"""
	tf_idf_mean = dict()
	# N = Number of movies
	N = len([tf_idf.index])
	for character in tf_idf.index:
		li = []
		for movie in tf_idf.columns:
			if str(tf_idf.loc[character][movie]) != 'nan':
				# Appending tf-idf values of character to a list
				li.append(tf_idf.loc[character][movie])
		# Updating two kinds of values: avg tfidf over their movies only and avg tfidf over entire mcu
		tf_idf_mean[character] = [np.mean(li), np.sum(li)/23]
	# Creating a data frame to be returned
	tf_idf_mean_df = pd.DataFrame.from_dict(tf_idf_mean).transpose()
	tf_idf_mean_df.columns = ['TF-IDF Their Movies Only', 'TF-IDF MCU Contribution']
	# Sorting based on character's contribution to the MCU
	tf_idf_mean_df = tf_idf_mean_df.sort_values('TF-IDF MCU Contribution', ascending=False)
	return tf_idf_mean_df
	
	
def generate_sheets(time_matrix, tf_idf_character, tf_idf_mcu):
	"""Method that generates CSV files of the results"""
	time_matrix.to_csv('character_screen_time_every_movie.csv')
	tf_idf_character.to_csv('character_tf_idf_every_movie.csv')
	tf_idf_mcu.to_csv('character_tf_idf_mcu_contribution.csv')


if __name__ == '__main__':
	# Input IMDB url containing the data
	url = 'https://www.imdb.com/list/ls066620113/?sort=list_order,asc&st_dt=&mode=detail&page=1&ref_=ttls_vm_dtl'
	# Fetching relevant data using web scraping
	movie_lengths, imdb_data = fetch_data(url)
	# Fetching a dict containg time for evry character over every movie
	char_movie_time_index = clean_data(imdb_data)
	# Making a matrix of the data after removing minor characters
	matrix = remove_characters(char_movie_time_index)
	# Calculating IDF values for each character
	idf_values = calculate_idf(matrix)
	# Converting time values in the matrix to float
	converted_matrix = convert_time_to_mins(matrix)
	# Calculating TF values for each character over each movie
	tf_values = calculate_tf(converted_matrix, movie_lengths)
	# Calculating TF-IDF values for each character over every movie
	tf_idf_values = calculate_tf_idf(tf_values, idf_values)
	# Calculating contribution of each character to their movies as well as to the entire MCU
	tf_idf_mcu_values = calculate_mean_tf_idf(tf_idf_values)
	# Generating CSV files with values
	generate_sheets(converted_matrix, tf_idf_values, tf_idf_mcu_values)
