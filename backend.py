

import re
import string
import math
from math import log
from dateutil import parser
from threading import Thread
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

from time import time
from timeit import timeit
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""### Text Preprocess

Notes about method:
* If name entity - keep original, save both for lemmatization and posting list in original form
* All else - lower text and replace abbriviations
* This means query should not be corrected for capitals, maybe just ass to query a first lower cased word if suspect it's an auto grammer corrector. check if Nir's queries are small letters or regular
"""

lemmatizer = WordNetLemmatizer()

"""### Text Preprocess"""

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = [
    'category', 'references', 'also', 'links', 'extenal', 'see',"links",
                    "may", "first","history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", '.', ',', '?', '!', ':', ';', '/', '\\', '-', '"', "'", "(", ")",
    "[", "]", "{", "}", "|", "*", "+", "@", "^", "&", "%", "#", "''",'``','...', '',' ', None
]
corpus_stopwords = set(corpus_stopwords + list(string.punctuation))

ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)
lemmatizer = WordNetLemmatizer()
"""
===============================================================================
Text preprocessing
===============================================================================
"""
def replace_contractions(token):
    """
    Replace common contractions like "n't" with their full words.

    Parameters:
        tokens (list of str): List of tokens.

    Returns:
        list of str: List of tokens with contractions replaced.
    """
    contractions_mapping = {
        "n't": "not",
        "'nt": "not",
        "'ll": "will",
        "'ve": "have",
        "'re": "are",
        "'d": "would",
        "'m": "am",
        ".": "",
        "'s": "is"
    }

    for key in contractions_mapping.keys():
        if key in token:
            i = token.index(key)
            if key == ".":
                return [token[:i], token[i+1:]]
            return [token[:i], contractions_mapping[key]]
    return [token]


def ParseDateFromToken(token):
    def is_likely_date(token):
        # Check for delimiters typically found in numeric shaped dates
        if re.search(r'(\d{1,4}[- /.]\d{1,2}[- /.]\d{1,4})', token):
            return True
        return False

    if not is_likely_date(token):
        return None

    try:
        parsed_date = parser.parse(token, ignoretz=True)

        # Validate year range to prevent OverflowError
        if parsed_date.year < 1 or parsed_date.year > 9999:
            return None

        # Further validation to ignore times without explicit dates
        if parsed_date.hour != 0 or parsed_date.minute != 0 or parsed_date.second != 0:
            # This means the token was more like a time than a date
            return None

        year = parsed_date.year
        full_month_name = parsed_date.strftime("%B")
        return str(year), full_month_name
    except (ValueError, OverflowError, TypeError):
        return None

def PreProcessText(text):
    '''
    Parse input text. Titles - all lowered, body - keeps capitalization under conditions.
    Out is a final list of lemmatized tokens.
    '''
    in_tokens = word_tokenize(text)

    tok_tmp = []
    for token in in_tokens:
      tmp = replace_contractions(token)
      tok_tmp.extend(tmp)

    tokens = [token.lower().strip(string.punctuation) for token in tok_tmp if token not in ALL_STOPWORDS]
    tok_tmp = []
    for token in tokens:
        date_data = ParseDateFromToken(token)  # list of 2 or None
        if date_data is not None:
            tok_tmp.extend(date_data)

    tokens.extend(tok_tmp)
    return tokens

def Extend_Query(tokens):
  '''
  Function extends query (specifically entity words) with synonims.
  Input are the initial tokens from the split action.
  '''
  def get_primary_synonyms(word):
    synonyms = set()
    synsets = wn.synsets(word)
    if synsets:
        primary_synset = synsets[0]
        synonyms.update(lemma.name() for lemma in primary_synset.lemmas())
    synonyms = list(synonyms)

    # Split to singular words:
    final_synonyms = []
    for term in synonyms:
      tmp = term.split('_')
      final_synonyms.extend(tmp)
    return final_synonyms

  tokens_final = []
  for token in tokens:
    ex = get_primary_synonyms(token)
    if len(ex) > 0:
      tokens_final.extend(ex)
  return set([token.lower() for token in tokens_final if token not in ALL_STOPWORDS])

def possible_containing_entity(query):
    '''
    Based on the existence of mid-sentence capitalization or number
    '''
    query = query[1:]
    # Regular expression pattern to match capitalized words or words containing numbers not at the beginning of the query
    entity_pattern = r'\b[A-Z0-9][a-z0-9]*\b'
    # Find all matches of the pattern in the query
    matches = re.findall(entity_pattern, query)
    # Return True if there are any matches, False otherwise
    return bool(matches)

def EntitiesKeywords(query):
    """
    Assuming there might be an entity in this query, try to get it.
    Will be used to calculate similarity score only for tokens with capital letters or numbers (additional similarity score)
    and to expand the query with important terms.
    """
    # Split the query into tokens
    tokens = word_tokenize(query)
    # lower first char and determine if first word needs to be included
    first = tokens[0][0].lower()
    # Extract tokens with capital letters or numbers
    entity_tokens = [token for token in tokens[1:] if any(char.isupper() or char.isdigit() for char in token)]
    if any(char.isupper() or char.isdigit() for char in first):
      entity_tokens.append(tokens[0])
    return ' '.join(entity_tokens)

"""### Retrieval Functions"""

def PreprocessQuery(query, expand = True, synonims = True, expansion_param = 2):
  '''
  Query preprocess is similar to body's - keeps mid query capitalization in an attempt to catch entities.
  In addition, adds lowered version of the capitalized word so that will be looked for too
  (in the title, which cannot keep the capitalization).
  Synonims are looked for only for entities
  '''
  extension = []
  # Tokenize the text
  if expand:
    original_tok = PreProcessText(query)
    if possible_containing_entity(query) or len(original_tok) == 1:
      entities = EntitiesKeywords(query)
      query = query + ' ' + (entities+' ') * (expansion_param - 1)
      if synonims:
        # If query has 1 word, treat as entity
        if len(original_tok) == 1:
          extension = Extend_Query(original_tok)
        else:
          # Expand only entities
          extension = Extend_Query(entities.split())
  tokens = PreProcessText(query)
  tokens.extend(extension)
  # For query, lemmatization only happens after expantion
  tokens = [lemmatizer.lemmatize(token) for token in tokens]
  return tokens


'''
================================================================================
Documents retrival from index
================================================================================
'''
'''
================================================================================
For Space and calculation efficiency, pass retrieval methods only necessary data
================================================================================
'''

def _get_postings(term, method, inverted, BUCKET_NAME, PageRank, PageViews):
    '''
    This function is a worker function desined to retrive and design a data structure from posting for 1 term.
    Will be activated through ThreadPool in order to minimize IO time.
    '''
    docs = inverted.read_a_posting_list('', term, BUCKET_NAME)
    index, idf = inverted.term_data[term]
    d = []

    for doc in docs:
        if doc[0] not in inverted.additional_info:
            continue
        pageviews = log(PageViews.get(doc[0], 0) +1 )  # limit values growth
        pagerank = log(PageRank.get(doc[0], 0) + 1)  # limit values growth
        info = inverted.additional_info[doc[0]]

        if method == 'Cosine':
            title = info[0]
            data = (doc[0], title, pagerank, pageviews)
        elif method == 'BM25':
            title, doc_size = info[0], info[1]
            data = (doc[0], title, doc[1], doc_size, pagerank, pageviews)

        d.append(data)

    if method == 'Cosine':
        return (term, index, idf), d
    elif method == 'BM25':
        return (term, idf), d



def RetrieveFromPosting(query_tokens, inverted, BASE_DIR, BUCKET_NAME, PageRank, PageViews, method='BM25'):
    '''
    ThreadPool wrapper designed to get the postings for the entire query (query_tokens).
    method parameter exists because for memory efficiency, different methods require different structure.
    '''
    fetched_docs = {}
    query_tokens = list(set([token for token in query_tokens if token in inverted.df]))  # Ensure uniqueness and filter irrelevant tokens

    with ThreadPoolExecutor(max_workers=len(query_tokens)) as executor:
        futures = [executor.submit(_get_postings, term, method, inverted, BUCKET_NAME, PageRank, PageViews) for term in query_tokens]
        for future in futures:
            key, value = future.result()
            fetched_docs[key] = value

    return fetched_docs

'''
================================================================================
Similarity functions
================================================================================
'''

def CosineSimilarity(fetched_docs, doc_vectors, query_tokens ,threshold=0.1):
  '''
  In classic CosineSimilarity we'll take a corpus size vector with entries for term in query and DOT PRODUCT it with corpus size vector for document
  with entries for terms and appearences in the doc. In order to be computationally efective we created a sparse vector for both query (a vector of [idf1, idf2,...] for term in
  parsed query), and for doc with [tf-idf1, tf-idf2,...] for document by term index.
  Input fetched docs data is shaped like {(term, index, idf): [(doc_id, title, pagerank, pageviews)]}, while the doc vector is being called from the inverted index.
  query_tokens are passed to allow highligting of important words (duplicates) in the similarity calculation.
  This function allows filtration of low similarity documents to ease of later calculations.
  '''
  # Assume query_terms is a list of terms in the processed query
  query_len = len(query_tokens)
  query_tf = {token: count / query_len for token, count in Counter(query_tokens).items()}
  query_vector = {key[1]: key[2]*query_tf[key[0]] for key in fetched_docs.keys()}  # Build query Vector {index: tf*idf}

  def cosine_similarity(query_vec, doc_vec):
      # Convert sparse vectors to NumPy arrays for efficient computation
      indices = list(set(query_vec.keys()) | set(doc_vec.keys()))
      q_vec = np.array([query_vec.get(index, 0) for index in indices])
      d_vec = np.array([doc_vec.get(index, 0) for index in indices])

      dot_product = np.dot(q_vec, d_vec)
      magnitude = np.linalg.norm(q_vec) * np.linalg.norm(d_vec)

      return 0 if magnitude == 0 else dot_product / magnitude

  # For each document, calculate its vector and then its cosine similarity with the query vector
  doc_data = {}
  for term_info, docs in fetched_docs.items():
    for doc_detail in docs:
        doc_id, title, pagerank, pageviews = doc_detail
        doc_vec = doc_vectors.get(doc_id, {})
        score = cosine_similarity(query_vector, doc_vec)
        if score >= threshold:
          doc_data[doc_id] = [title, cosine_similarity(query_vector, doc_vec), pagerank, pageviews]

  return doc_data



def BM25_Similarity(fetched_docs, AVG_doc_length, query_tokens, k1=1.5, b=0.75, threshold=0.1):
    """
    Calculates the BM25 similarity scores for a collection of documents based on the terms they contain.

    Args:
    fetched_docs (dict): A dictionary where each key is a tuple containing term information (term, df, idf, term_total)
                         and each value is a list of tuples containing document information
                         (doc_id, title, tf, doc_length, norm_tf, tf-idf, pagerank).
                         Example structure:
                         {(term, idf): [(doc_id, title, tf, doc_length, pagerank, pageview), ...], ...}

    AVG_doc_length (float): The average document length across the corpus.

    k1 (float): The BM25 k1 parameter, controlling the document term frequency scaling. Typically in [1.2, 2.0].

    b (float): The BM25 b parameter, controlling the scaling by document length. Typically set to 0.75.

    Returns:
    list: A sorted list of tuples, where each tuple contains (doc_id, (title, score, pagerank, pageview)), with 'score'
          being the BM25 score. The list is sorted by score in descending order.
    """
    def calculate_bm25(tf, doc_length, AVG_doc_length, idf, k1=1.5, b=0.75):
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / AVG_doc_length))
        return idf * (numerator / denominator)

    # Initialize a dictionary to hold document scores
    doc_scores = {}
    query_len = len(query_tokens)
    query_tf = {token: count / query_len for token, count in Counter(query_tokens).items()}  # For keyword importance

    for term_info, docs in fetched_docs.items():
        term, idf = term_info
        for doc in docs:
            doc_id, title, tf, doc_length, pagerank, pageviews = doc
            score = calculate_bm25(tf, doc_length, AVG_doc_length, idf, k1, b) * query_tf[term]  # Incorporate term significance

            # Initialize or update the doc score
            if doc_id not in doc_scores:
                doc_scores[doc_id] = [title, score, pagerank, pageviews]
            else:
                # Update score by adding the new score to the existing one
                doc_scores[doc_id][1] += score

    # Filter documents by threshold after scores are fully computed
    doc_scores = {doc_id: data for doc_id, data in doc_scores.items() if data[1] >= threshold}

    return doc_scores


# """* Change back base dir in threads"""
def Parallel_Search(query, invertedT, invertedB,BASE_DIR_T,BASE_DIR_B, BUCKET_NAME, PageRank, PageViews):
    def merge_thread_outputs(dict1, dict2):
        """
        Merges outputs from two dictionaries into a single dictionary where each key (doc_id) maps to
        a list that includes all similarity scores along with title, pagerank, and pageview.

        Parameters:
        - dict1: Dictionary from the first thread with doc_id as keys and
                 [title, similarity_score, pagerank, pageview] as values.
        - dict2: Dictionary from the second thread with doc_id as keys and
                 [title, similarity_score, pagerank, pageview] as values.

        Returns:
        - merged_dict: A dictionary with merged results. For each doc_id, the value is a list:
          [title, similarity_score_1, similarity_score_2, pagerank, pageview].
        """

        merged_dict = {}

        # Merge dict1 and dict2 based on dict 2 (possibly more keys)
        for doc_id, values in dict2.items():
            title, similarity_s_2, pagerank, pageview = values
            similarity_s_1 = dict1.get(doc_id, [None, 0, None, None])[
                1]  # Get similarity_s_1 if exists in dict2, else 0

            merged_dict[doc_id] = [title, similarity_s_1, similarity_s_2,
                                   pagerank, pageview]

        # Handle doc_ids unique to dict1
        for doc_id, values in dict1.items():
            if doc_id not in merged_dict:
                title, similarity_s_1, pagerank, pageview = values
                similarity_s_1 = 0
                merged_dict[doc_id] = [title, similarity_s_1, similarity_s_2,
                                       pagerank, pageview]

        return merged_dict

    def worker_theards(query_tokens, inverted_index, BASE_DIR, similarity_method, output_dict, key_suffix):
        """
        Thread worker function to search a single index.
        """
        if similarity_method == 'BM25':
            fetched_docs = RetrieveFromPosting(query_tokens, inverted_index, BASE_DIR, BUCKET_NAME, PageRank, PageViews, method=similarity_method)
            fetched_docs = BM25_Similarity(fetched_docs, inverted_index.AVG_doc_length, query_tokens, 1.5, 0.75, 0.73)
        elif similarity_method == 'Cosine':
            fetched_docs = RetrieveFromPosting(query_tokens, inverted_index, BASE_DIR, BUCKET_NAME, PageRank, PageViews, method=similarity_method)
            fetched_docs = CosineSimilarity(fetched_docs, inverted_index.doc_vectors, query_tokens, 0.25)
        else:
            raise ValueError("Unsupported similarity method: {}".format(similarity_method))

        output_dict[key_suffix] = fetched_docs

    def GetWeights(query):
        '''
        Create weights for the different cases in the
        '''
        query_length = PreprocessQuery(query, False, False)
        if query_length == 1:
            return 0.2, 0.2, 0.2, 0.4
        else:
            if possible_containing_entity(query):
                return 0, 0.8, 0, 0.2
            else:
                return 0.2, 0.4, 0.2, 0.2

    def CustomMinMaxScaling(df):
        """
        Perform Min-Max scaling on specified columns of the DataFrame using provided min-max values.

        Parameters:
        df (DataFrame): The DataFrame containing the data to be scaled.
        min_max_values (dict): A dictionary containing the minimum and maximum values for each column.
        columns (list): List of column names to be scaled.
        num_threads (int): Number of threads to use for parallel processing. If None, uses the number of available CPUs.

        Returns:
        DataFrame: DataFrame with specified columns scaled using Min-Max scaling.
        """
        min_max_values = {
            'Similarity_Score_Title_Index': (0, 1.0000000000000002),
            'Similarity_Score_Body_Index': (
            0.2500000344706341, 8.205944171811298),
            'PageRank': (0.0, 9.201776686284328),
            'PageViews': (0.0, 19.014704766035365)}
        columns = list(min_max_values.keys())
        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Define a function for scaling each column
            def scale_column(col):
                min_val, max_val = min_max_values[col]
                return (df[col] - min_val) / (max_val - min_val)

            # Submit tasks to the executor for each column
            scaled_columns = executor.map(scale_column, columns)

            # Wait for all tasks to complete and retrieve the results
            scaled_columns = list(scaled_columns)

        # Create a new DataFrame with the scaled columns
        scaled_df = pd.DataFrame({col: scaled_col for col, scaled_col in
                                  zip(columns, scaled_columns)})

        return scaled_df



    def _search(query, invertedT, invertedB,BASE_DIR_T,BASE_DIR_B) :
        """
        Performs a search query across two indexes using a specified similarity method. Results are merged
        and returned as a pandas DataFrame. Utilizes a Thread per retrieval task in order to be time efficient.

        Parameters:
        - query (str): The search query.
        - inverted1, inverted2: Inverted index objects with term data and AVG_doc_length.
        - similarity_method (str): The similarity method to use ('BM25' or 'Cosine').

        Returns:
        - DataFrame: A pandas DataFrame with columns ['doc_id', 'title', 'Similarity_Score_Title_Index',
                      'Similarity_Score_Body_Index', 'pagerank', 'pageviews'].
        """
        query_tokens = PreprocessQuery(query, expand = True, synonims = False, expansion_param = 2)
        if len(query_tokens) == 0:
            return [("404", "Could not find any relevant pages")]
        results = {}
        threads = []

        # Create threads for each index + entity similarity
        threads.append(Thread(target=worker_theards, args=(query_tokens, invertedT, BASE_DIR_T, 'Cosine', results, 'Title Search')))
        threads.append(Thread(target=worker_theards, args=(query_tokens, invertedB, BASE_DIR_B, 'BM25', results, 'Body Search')))

        # Start threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Assuming both indexes return documents in the same format, merge outputs
        merged_docs = merge_thread_outputs(results['Title Search'], results['Body Search'])

        # Convert to DataFrame
        data = []
        for doc_id, details in merged_docs.items():
            # Assuming each similarity method returns: (title, score, pagerank, pageviews)
            data.append((doc_id, *details))

        fetched_docs = pd.DataFrame(data, columns=['Doc_ID', 'Title', 'Similarity_Score_Title_Index', 'Similarity_Score_Body_Index', 'PageRank', 'PageViews'])

        features = ['Similarity_Score_Title_Index',
                    'Similarity_Score_Body_Index',
                    'PageRank', 'PageViews']

        fetched_docs['Query Length'] = len(
            PreprocessQuery(query, False, False))
        fetched_docs['Contains Entity'] = 1 if possible_containing_entity(
            query) else 0
        fetched_docs[features] = CustomMinMaxScaling(fetched_docs)
        w1, w2, w3, w4 = GetWeights(query)
        fetched_docs['Weighted_Score'] = w1 * fetched_docs[
            'Similarity_Score_Title_Index'] + \
                                         w2 * fetched_docs[
                                             'Similarity_Score_Body_Index'] + \
                                         w3 * fetched_docs['PageRank'] + \
                                         w4 * fetched_docs['PageViews']

        docs = fetched_docs.sort_values(by='Weighted_Score', ascending=False)[:30]
        doc_id_title_tuples = [(str(doc_id), title) for doc_id, title in zip(docs['Doc_ID'], docs['Title'])]

        return doc_id_title_tuples






    return _search(query, invertedT, invertedB,BASE_DIR_T,BASE_DIR_B)





"""
================================================================================
ENV Parameters
================================================================================
"""

from google.cloud import storage
import pickle
from inverted_index_gcp import *

# Initialize a client
client = storage.Client()

def read_pkl_from_gcs(bucket_name, file_name):
    """Read a .pkl file from a Google Cloud Storage bucket."""
    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(file_name)

    # Download the file to a local temporary file
    temp_file_path = "/tmp/temp_file.pkl"  # You can change the path as needed
    blob.download_to_filename(temp_file_path)

    # Load the data from the pickle file
    with open(temp_file_path, "rb") as f:
        data = pickle.load(f)

    return data


BUCKET_NAME = 'shahar_ir_project'
INDEX_NAME_T = 'Index_Title_Final_Corrected'
BASE_DIR_T = 'Index_Title_Final_Corrected'
INDEX_NAME_B = 'Index_Body_Final_Corrected'
BASE_DIR_B = 'Index_Body_Final_Corrected'
PageViews = read_pkl_from_gcs(BUCKET_NAME, f'BigDataFiles/pageviews-202108-user.pkl')
PageRank = dict(read_pkl_from_gcs(BUCKET_NAME, f'BigDataFiles/PageRank_dict.pkl'))
index_title = read_pkl_from_gcs(BUCKET_NAME, f'{BASE_DIR_T}/{INDEX_NAME_T}.pkl')
index_body = read_pkl_from_gcs(BUCKET_NAME, f'{BASE_DIR_B}/{INDEX_NAME_B}.pkl')

'''
Add these manually to backend file
'''

index_title.N = 6348910
Body_additional_info = read_pkl_from_gcs(BUCKET_NAME, f'{BASE_DIR_B}/Additional_Info.pkl')
Body_term_data = read_pkl_from_gcs(BUCKET_NAME, f'{BASE_DIR_B}/TermData.pkl')
index_body.term_data = Body_term_data
index_body.additional_info = Body_additional_info
def search_for_fronted(query):
    return Parallel_Search(query, index_title, index_body,BASE_DIR_T,BASE_DIR_B, BUCKET_NAME, PageRank, PageViews)