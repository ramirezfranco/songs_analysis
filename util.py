import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from pycorenlp import StanfordCoreNLP
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
import urllib3
import bs4
import lyricsgenius
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from os import path
import gensim
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from flask import render_template
import flask
from IPython.display import HTML, display
# from googleapiclient.discovery import build
import creds


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

nlp = StanfordCoreNLP('http://localhost:9000/')
'''
Additionally, run the following java code in the command promt 
in the directory where stanford CoreNLP is located:
'java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000'
For more details see: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
'''

# Do not forget to get and use your own credentials
genius = lyricsgenius.Genius(creds.access_token)


def make_soup(myurl):
	'''
	Creates a BeautifulSoup object
	inputs:
		myurl - string with the url
	output:
		BeautifulSoup object
	'''
	pm = urllib3.PoolManager()
	html = pm.urlopen(url = myurl, method = 'GET', redirect= False).data
	return bs4.BeautifulSoup(html, "lxml")

def quick_clean_doc(text, language='spanish'):
	'''
	Removes unknown characters and punctuation, change capital to lower letters and remove
	stop words. If stem=False
	Inputs:
	sentence (string): a sting to be cleaned
	Returns: a string
	'''
	#tokens = regex_tokenizer.tokenize(text)
	tokens = nltk.word_tokenize(text)
	tokens = [t.lower() for t in tokens]
	tokens = [t for t in tokens if t not in stopwords.words(language)+[p for p in string.punctuation]]
	return ' '.join(tokens)


def clean_tokens_doc(text, language='spanish'):
	'''
	Removes unknown characters and punctuation, change capital to lower letters and remove
	stop words. If stem=False
	Inputs:
	sentence (string): a sting to be cleaned
	Returns: a string
	'''
	#tokens = regex_tokenizer.tokenize(text)
	tokens = nltk.word_tokenize(text)
	tokens = [t.lower() for t in tokens]
	tokens = [t for t in tokens if t not in stopwords.words(language)+[p for p in string.punctuation]]
	return tokens


def songs_dictionary(artist_name):
	artist = genius.search_artist(artist_name)
	if artist:
		songs = artist.songs
		songs_dict = {
			i:{
				'title': songs[i].title, 
				'album': songs[i].album, 
				'year': songs[i].year, 
				'lyrics': songs[i].lyrics, 
				'artist': songs[i].artist
			} 
			for i in range(len(songs))
		}
		return songs_dict
	


def raw_corpus(songs_dict):
	corpus = [v['lyrics'] for v in songs_dict.values()]
	return corpus


def cleaned_corpus(songs_dict):
	corpus = [quick_clean_doc(v['lyrics']) for v in songs_dict.values()]
	return corpus

def get_ngrams(text, n):
	tokens = nltk.word_tokenize(text)
	n_grams = [', '.join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]
	return n_grams

def ngram_freq(corpus, size):
	'''
	Computes the frequency of n-grams according to size and
	retuns an ordered data frame.
	Inputs:
		corpus (string): text to be analized
		size (int): size of n-grams
	Returns: a data frame
	'''
	frequencies = {}
	n_grams = [get_ngrams(doc, size) for doc in corpus]
	ng_corp = []
	
	for doc in n_grams:
		ng_corp+=doc

	for ng in ng_corp:
		if ng not in frequencies.keys():
			frequencies[ng] = 1
		else:
			frequencies[ng] += 1

	freq_list = [(k, v) for k, v in frequencies.items()]
	df = pd.DataFrame(freq_list, columns=[str(size)+'-gram', 'Frequency'])
	df =  df.sort_values(by='Frequency', ascending=False)
	return df.reset_index().drop('index', axis=1)

def entity_mentions(sentence):
	'''
	Gets a list for a dictionary for every token that has an entity tag.
	Inputs: 
		- Sentence(str): a text
	Returns a list of dictionaries
	'''
	#sentences = box.get_content(file)
	annotations_list = []
	annotations = nlp.annotate(sentence, properties={'annotators': 'ner', 'outputFormat': 'json'})
	if type(annotations) == dict and 'sentences' in annotations.keys():
		for i in range(len(annotations['sentences'])):
			annotations_list += annotations['sentences'][i]['entitymentions']
	return annotations_list

def ners_in_corpus(corpus):
	all_ners = {}
	for document in corpus:
		ner_doc = entity_mentions(document)
		for mention in ner_doc:
			if mention['ner'] not in all_ners.keys():
				all_ners[mention['ner']] = [mention['text']]
			else:
				all_ners[mention['ner']] += [mention['text']]
	return all_ners


def ners_sentences(corpus, tag_list):
	all_sentences = {tag:[] for tag in tag_list}
	for document in corpus:
		annotations = nlp.annotate(document, properties={'annotators': 'ner', 'outputFormat': 'json'})
		if type(annotations) == dict and 'sentences' in annotations.keys():
			annotations = annotations['sentences']
			sentences_dict = {
				' '.join([token['originalText'] for token in sentence['tokens']]):[ner['ner'] for ner in sentence['entitymentions']] for sentence in annotations}
			for sent, ners in sentences_dict.items():
				for k in all_sentences.keys():
					if k in ners:
						all_sentences[k].append(sent)
	return all_sentences
			


def count_dict(a_list):
	frequency ={}
	for element in a_list:
		if element.lower() not in frequency.keys():
			frequency[element.lower()] = 1
		else:
			frequency[element.lower()] += 1
	return frequency


def ners_frequency(ners_dict, tag, top=None):
	info = count_dict(ners_dict[tag])
	df = pd.DataFrame.from_dict(info, orient='index', columns = ['Frequency'])
	if not top:
		return df.sort_values(by='Frequency', ascending=False)
	else:
		return df.sort_values(by='Frequency', ascending=False)[:top]



def similar(word, model):
	result = model.most_similar(word)
	results = [t[0] for t in result]
	return results


def analogy(x1, x2, y1, model):
	result = model.most_similar(positive=[y1, x2], negative=[x1])
	results = [t[0] for t in result]
	return results

app = flask.Flask('my app')

def compare_embedings(models_dict, querty):
	names = list(models_dict.keys())
	pool = [similar(querty, models_dict[name]) for name in names]
	results = [['Rank']+names]
	for i in range(len(pool[0])):
		results.append([i+1]+[l[i] for l in pool])
	return results

def present_results(querty, results_list):
	with app.app_context():
		table = render_template(
        	'compare_table_alt.html',
        	word = querty,
        	columns = results_list[0],
        	data = results_list[1:])
	return display(HTML(table))

def compare_models(models_dict, querty):
	table = compare_embedings(models_dict, querty)
	return present_results(querty, table)

def create_image(imgpath):
	np_image = np.array(Image.open(path.join(os.getcwd()+imgpath)))
	return np_image


def create_cloud_image(imgpath, df_freq, title):
	font = {
		'family': 'serif',
		'color':  'darkblue',
		'weight': 'normal',
		'size': 16,
		}
	
	mask = create_image(imgpath)
	explanation = 'Elaboracion: Jesus I. Ramirez Franco'
	wcf = WordCloud(background_color="White", max_words=1000, mask=mask,
			   max_font_size=90, random_state=42, repeat=True)
	wcf.generate_from_frequencies(df_freq)
	image_colors = ImageColorGenerator(mask)
	plt.figure(figsize=[15,15])
	plt.text(50, 200, title,
		 fontdict=font)
	plt.text(1000, 1000, explanation)
	plt.imshow(wcf.recolor(color_func=image_colors), interpolation="bilinear")
	plt.axis("off")
	plt.show()


def create_cloud(df_freq, title, colors):
	font = {
		'family': 'serif',
		'color':  (.18, .19, .43),
		'weight': 'bold',
		'size': 15,
		'fontstyle':'normal',
		'fontname': 'arial'
		}
	x, y = np.ogrid[:300, :300]
	mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
	mask = 255 * mask.astype(int)
	wcsb = WordCloud(background_color="White", max_words=1000, random_state=42, colormap=colors, repeat=True, mask=mask, contour_color='steelblue')
	wcsb.generate_from_frequencies(df_freq)
	plt.figure(figsize=(6,6)) 
	plt.imshow(wcsb, interpolation='bilinear')
	plt.title(title,loc='center', fontdict=font)
	explanation = 'Jesus I. Ramirez Franco'
	plt.text(180, 300, explanation)
	plt.axis("off")
	plt.show()


#def translate_list(list):



def create_eta(priors, etadict, ntopics):
	'''
	Creates an eta matrix to specify the important terms that a topic most contain.
	Inputs:
		- priors (dict): dictionary where every key is a term (str) and every value 
		  is the number of topic (int) where the term most appear.
		- etadic (dict): dictionary produced by the training corpus.
		- ntopics (int): number of topics in the model.
	'''
	eta = np.full(shape=(ntopics, len(etadict)), fill_value=1) # create a (ntopics, nterms) matrix and fill with 1
	for word, topic in priors.items(): # for each word in the list of priors
		keyindex = [index for index,term in etadict.items() if term==word] # look up the word in the dictionary
		if (len(keyindex)>0): # if it's in the dictionary
			eta[topic,keyindex[0]] = 1e7  # put a large number in there
	eta = np.divide(eta, eta.sum(axis=0)) # normalize so that the probabilities sum to 1 over all topics
	return eta

def make_LDA(vec_corpus, dictionary, n, prior='auto', iter_v=200, pass_v=150, decay_v=0.7):
	'''
	Creates an LDA model with the given parameters.
	Inputs:
		- vec_corpus: Stream of document vectors or sparse matrix.
		- dictionary (dict): Mapping from word IDs to words.
		- n (int): The number of requested latent topics to be extracted from the 
		  training corpus.
		- prior (dict): Map from terms to topics.
		- iter_v (int): number os maximum iterations.
		- pass_v (int): number of passes through the corpus.
		- deccay_v (float): A number between (0.5, 1].
	'''
	if prior =='auto':
		eta_matrix = 'auto'
	else:
		eta_matrix = create_eta(prior, dictionary, n)
	model = gensim.models.ldamodel.LdaModel(
		corpus=vec_corpus, 
		id2word=dictionary, 
		num_topics=n,
		random_state=42, 
		eta=eta_matrix,
		iterations=iter_v,
		eval_every=-1, 
		update_every=1,
		passes=pass_v, 
		alpha='auto', 
		per_word_topics=True,
		decay = decay_v
	)
	return model

def evaluate_model(model, corpus, dictionary, n):
	'''
	Computes the perplexity, coherence and topics of an LDA model.
	Inputs:
		- model: An LDA model produced using Gensim.
		- corpus: The corpus used to train the model.
		- dictionary: the dictionary created by the corpus.
		- n: number of topics of the model
	'''
	perplexity = model.log_perplexity(corpus)
	topics = [[dictionary[w] for w,f in model.get_topic_terms(topic, 10)] for topic in range(n)]
	cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
	coherence = cm.get_coherence()
	return topics, perplexity, coherence

def dif_models(vec_corpus, dictionary, n_list, iter_list, pass_list, decay_list, prior="auto"):
	'''
	Computes different LDA models and preserve the one with the best coherence.
	Inputs:
		- vec_corpus: the corpus used to train the model.
		- dictionary: the dictionary created by the corpus
		- n_list: list of different numbers of topics.
		- iter_list: list of different numbers of iterations values.
		- pass_list: list of different pass values.
		- decay_list: list of different decay values.
		- prior (dict): dictionary where every key is a term (str) and every value 
		  is the number of topic (int) where the term most appear.
	'''
	best_model = None
	best_coherence = -10000
	best_perplexity = -10000
	topics_bm = None
	results = []
	for n in n_list:
		for i in iter_list:
			for p in pass_list:
				for d in decay_list:
					lda = make_LDA(vec_corpus, dictionary, n, prior, i, p, d)
					topics, perp, cohe = evaluate_model(lda, vec_corpus, dictionary, n)
					results.append([n, i, p, d, perp, cohe])
					if cohe > best_coherence:
						best_model = lda
						topics_bm = topics
						best_coherence = cohe
						best_perplexity = perp
	colum_names = ['Number of Topics', 'Iterations', 'Passes', 'Decay', 'Perplexity', 'Coherence']
	df = pd.DataFrame(results, columns=colum_names)
	df = df.sort_values(by='Coherence', ascending=False)
	return df, best_model, topics_bm

def print_topics(topics):
	'''
	Print the top 10 most important terms of every topic of an LDA model.
	Inputs:
		- topics (list): list of lists of terms by topic, produced by an LDA model.
	'''
	for topic in range(len(topics)):
		print('Topic ', topic)
		print(', '.join(topics[topic]))
		print('**************************************************************************************************')


def doc_topic_matrix(model, corpus_vec):
	doc_topics = model.get_document_topics(corpus_vec)
	doc_topics_dict = {i:{t[0]:t[1] for t in doc_topics[i]} for i in range(len(doc_topics))}
	doc_topics_matrix = pd.DataFrame.from_dict(doc_topics_dict, orient='index').fillna(0)
	return doc_topics_matrix
	


