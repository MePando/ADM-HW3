# General
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Set, Iterable, Tuple, Any
from tqdm.auto import tqdm

# For Search Engine
from joblib import dump, load
from sortedcontainers import SortedSet, SortedList
from collections import Counter

import heapq

class Vocabulary():
  """A vocabulary of words present in pd.DataFrame columns.

  Args:
    - dataframe (pd.DataFrame): The pd.DataFrame from which to build the vocabulary.
    - columns (List[str]): Which columns of the pd.DataFrame to use, if None all columns are used. Default None.

  Attributes:
    - dataframe (pd.Dataframe): Where it is stored the dataframe input.
    - columns (List[str]): Where it is stored the columns input.
    - index2word (Dict[str, str]): A dictionary in which we have as keys the index and as values the words.
    - word2index (Dict[str, str]): A dictionary in which we have as keys the word and as values the corresponding numbers in the index2word index.
    - word2frequency (Dict[str, int]): A dictionary in which we have as keys words and as values the numbers of occurences of that word in the entire corpus.
    - num_words (int): Number of total words in the vocabulary.
    - maximum_frequency (int): The maximum frequency.
    - words_with_maxf (List[str]): The list of words with the maximum frequency.
  """
  def __init__(self,
               dataframe: pd.DataFrame,
               columns: List[str] = None):

    self.index2word = {}
    self.word2index = {}
    self.word2frequency = {}
    self.num_words = 0
    self.maximum_frequency = 0
    self.words_with_maxf = []

    self.columns = columns

    if self.columns is None:
      self.columns = list(dataframe.columns)
    
    self.dataframe = dataframe[self.columns]

    for column in tqdm(self.columns):
      for elem in self.dataframe[column]:
        for word in elem:
          self.add_word(word)

  def __getitem__(self,
                  item: Union[int, str]) -> Union[str, int]:
      """Override of the __getitem__ method:
      Given an int as index, it returns the word corresponding to that index.
      Given a str as word, it returns the index corresponding to that word.

      You could also use 'get_word(index)' method if your searching for a word given an index.
      You could also use 'get_index(word)' method if your searching for an index given a word.

      Args:
        - item (int / str): If int it is index of the word, if str it is the word.

      Returns:
        - str / int : if item is int, then it returns the corresponding word, while if item is str it returns the corresponding index for that word.
      """
      if isinstance(item, int):
        return self.index2word[item]
      elif isinstance(item, str):
        return self.word2index[item]
      else:
        raise Exception("Item is expected to be an integer if you're searching the corresponding word, or a str if you're searching the index")

  def __str__(self):
    """An override of the '__str__()' method, in this way we're able to print some informations about the vocabulary.

    Returns:
      - str : The str rappresentation of the vocabulary.
    """
    return f"Number of words in the vocabulary: {self.num_words}\nMaximum frequency value: {self.maximum_frequency} -> List of words with the maximum frequency: {self.words_with_maxf}"

  def __len__(self) -> int:
    """An ovverride of the '__len__()' method:

    Returns:
      - self.num_words (int): The number of words present in the Vocabulary.
    """
    return self.num_words

  def add_word(self,
               word: str) -> None:
    """A function used to add a word to our vocabulary.

    Args:
      - word (str): The word to add.
    
    Returns:
      - None
    """
    if word not in self.word2index:
      self.word2index[word] = self.num_words
      self.word2frequency[word] = 1
      self.index2word[self.num_words] = word
      self.num_words += 1

      if self.word2frequency[word] > self.maximum_frequency:
        self.maximum_frequency = self.word2frequency[word]
        self.words_with_maxf = [word]
      elif self.word2frequency[word] == self.maximum_frequency:
        self.words_with_maxf.append([word])

    else:

      self.word2frequency[word] += 1

      if self.word2frequency[word] > self.maximum_frequency:
        self.maximum_frequency = self.word2frequency[word]
        self.words_with_maxf = [word]
      elif self.word2frequency[word] == self.maximum_frequency:
        self.words_with_maxf.append(word)

    return None

  def get_word(self,
               index: int) -> str:
      """Given an int as index, it returns the word corresponding to that index.
      You could also use 'Vocabulary[index]'.

      Args:
        - index (int): The index of the word.

      Returns:
        - str : The word corresponding to that index.
      """
      return self.index2word[index]

  def get_index(self,
                word: str) -> int:
      """Given a str as a word, it returns the index corresponding to that word.
      You could also use 'Vocabulary[word]'.

      Args:
        - word (word): The word.

      Returns:
        - int : The index corresponding to that word.
      """
      return self.word2index[word]

  def get_frequency(self,
                    word: str) -> int:
    """It returns the corresponding frequency of the input word.

    Args:
      - word (str): A word in the vocabulary.

    Returns:
      - int : The corresponding frequency.
    """
    return self.word2frequency[word]
  
  def words(self):
    """Get the list of words present in the vocabulary.

    Returns:
      - List[str] : The list of all the words.
    """
    return list(self.word2index.keys())
  
  def sort_fun(self,
               x: str) -> int:
    """A function used to sort the SortedSet on the created index.

    Args:
      - x (str): The document.

    Returns:
      - int : The number code of that document.
    """
    return int(x.split("_")[1])

  def create_index(self,
                   path: str = None) -> Dict[int, SortedSet]:
    """A function that returns the index on the corpora of the self.dataframe using the vocabulary.

    Args:
      - path (str): The name of the pickle file where we save the index, if None the index is not saved. Default None.

    Returns:
      - index (Dict[int, SortedSet]): The index of the self.dataframe based on the vocabulary.
    """
    index = {}

    for word in tqdm(self.words()):
      index[self.word2index[word]] = SortedSet(key=self.sort_fun)
      for column in self.columns:
        index[self.word2index[word]].update(self.dataframe.index[self.dataframe[column].apply(lambda x: word in x)].tolist())

    if path:
      dump(index, path)

    return index

  def save(self,
           path: str) -> None:
    """Save the vocabulary as a pickle file.

    Args:
      - path (str): The name/path where to save the vocabulary.

    Returns:
      - None
    """
    dump(self, path)
    return None

  @classmethod
  def loader(self,
             path: str):
    """Load the vocabulary from a pickle file.

    Args:
      - path (str): The path to the picke file.
    """
    return load(path)



class Index():
  """An Index that has as keys words and as corresponding values SortedSets of the documents where each word is contained.

  Args:
    - index (Dict[int, Iterable[str]]): The dictionary to index.

  Attributes:
    - index (Dict[int, Iterable[str]]): A dictionary that has as key the term_id and as value the SortedSet of documents that contains that word. Default None.
    - num_words (int): The number of words in the index.
  """
  def __init__(self,
               index: Dict[int, Iterable[str]] = None):
    self.index = {}
    self.num_words = 0

    if index is not None:
      for term_id in tqdm(list(index.keys())):
        self.index[term_id] = SortedSet(index[term_id], key=self.sort_fun)
        self.num_words += 1

  def __setitem__(self,
                  term_id: int,
                  documents: SortedSet) -> None:
    """An override of the method __setitem__.

    Args:
      - word (str): The term_id to add in our dictionary.
      - documents (SortedSet): The SortedSet of documents in which word is present.
    """
    if not isinstance(term_id, int):
      raise Exception(f"The items in the index should be an int, got a: {type(term_id)}")
    if not isinstance(documents, SortedSet):
      raise Exception(f"The value for each item should be a SortedSet, got a: {type(documents)}")

    self.index[term_id] = documents
    self.num_words += 1

    return None

  def __getitem__(self,
                  term_id: int) -> SortedSet:
    """Override of the __getitem__ method:
    Given an int as term_id, it returns the index corresponding documents in which the term is present.

    Args:
      - term_id (int): The term_id of the word to search for.

    Returns:
      - SortedSet:  It returns the documents in which that word is present.
    """
    if not isinstance(term_id, int):
      raise Exception("Item is expected to be an int.")

    try:
      result = self.index[term_id]
    except:
      print(f"'{term_id}' not found in Index")
      result = SortedSet()

    return result

  def __str__(self) -> str:
    """An override of the '__str__()' method:

    Returns:
      - str : A string view of the Index.
    """
    return f"{self.index}"

  def __len__(self) -> int:
    """An override of the __len__ method.

    Returns
      - self.num_words (int): The number of words present in the Index.
    """
    return self.num_words
  
  @classmethod
  def sort_fun(self,
               x: str) -> int:
    return int(x.split("_")[1])

  def keys(self) -> List[int]:
    """Get the list of words present in the index.

    Returns:
      - List[int] : The list of all the words term_id.
    """
    return list(self.index.keys())

  def save(self,
           path: str) -> None:
    """To save an Index in a given path.

    Args:
      - path (str): The path where the index is going to be saved.

    Returns:
      - None
    """
    dump(self, path)
    return None
  
  @classmethod
  def loader(self,
             path: str) -> None:
    """The loader method used to load a saved Index.

    Args:
      - path (str): The path where the index to load is stored.

    Returns:
      - None
    """
    index = load(path)

    if not isinstance(index, Index):
      index = Index(index)

    return index



class SearchEngine():
  """A class that rappresent a Search Engine, here we find all the methods needed to query on the dataframe.

  Args:
    - dataframe (pd.DataFrame): The pd.DataFrame to query.
    - vocabulary (Vocabulary): The vocabulary of the dataframe.
    - index: The index word->documents of the dataframe.

  Attributes:
    - dataframe (pd.DataFrame): The variable in which we save the dataframe input.
    - vocabulary (Vocabulary): The variable in which we save the vocabulary input.
    - index (Index): The variable in which we save the index input. 
  """
  def __init__(self,
               dataframe: pd.DataFrame,
               vocabulary: Vocabulary,
               index: Index):

    print("Sanity Check between the Vocabulary and the Index:", end="\t")
    if not len(vocabulary) == len(index) :
      raise Exception(f"Inconsistency between the number of words in the vocabulary and the number of words in the index.\nVocabulary: {len(vocabulary)}\nIndex: {len(index)}")

    print("[PASSED]")

    self.index = index
    self.dataframe = dataframe
    self.vocabulary = vocabulary

  def words(self) -> List[str]:
    """Returns the list of the words.

    Returns:
      - List[tr]: The list of the words.
    """
    return self.vocabulary.words()

  def tf(self,
         word: str,
         document: str) -> float:
    """The function that calculates TF of a word and a document.

    Args:
      - word (str): The word for which we calculate the TF.
      - document (str): The document for which we calculate the TF.

    Returns:
      - float : The TF value for that word and document.
    """
    frq_t_in_d = 0
    for column in self.vocabulary.dataframe.columns:
      frq_t_in_d += self.vocabulary.dataframe[column][document].count(word)

    return 1.0 + math.log10(frq_t_in_d)

  def idf(self,
          word: str) -> float:
    """The function that calculates IDF of a word and a document.

    Args:
      - word (str): The word for which we calculate the IDF.

    Returns:
      - float : The IDF value for that.
    """
    n_docs_with_word = (len(self.index[self.vocabulary.get_index(word)]))

    return math.log10((1.0+len(self.vocabulary.dataframe))/(1.0 + n_docs_with_word))+1.0

  def tf_idf(self,
             word: str,
             document: str) -> float:
    """The function that calculates TF-IDF of a word and a document.

    Args:
      - word (str): The word for which we calculate the TF-IDF.
      - document (str): The document for which we calculate the TF-IDF.

    Returns:
      - float : The TF value for that word and document.
    """
    return self.tf(word, document)*self.idf(word)

  def create_index_tf_idf(self,
                          path: str = None) -> Dict[int, List[Tuple[str, float]]]:
    """A function that returns the index tf-idf on the corpora of the self.dataframe.

    Args:
      - path (str): The name of the pickle file where we save the index, if None the index is not saved. Default None.

    Returns:
      - index_tf_idf (Dict[int, List[Tuple[str, float]]]): The index with the tf-idf.
    """
    self.index_tf_idf = {}
    for word in tqdm(self.words()):
      self.index_tf_idf[self.vocabulary.get_index(word)] = []
      for doc in self.index[self.vocabulary.get_index(word)]:
        self.index_tf_idf[self.vocabulary.get_index(word)].append((doc, self.tf_idf(word, doc)))

    if path:
      dump(self.index_tf_idf, path)

    return self.index_tf_idf

  def tf_idf_query(self,
                   query: List[str]) -> Dict[str, float]:
    """A function to calculate the TF-IDF for a query.

    Args:
      - query List[str]: A query.

    Returns
      - tf_idf_q (Dict[str, float]): The dictionary for the query with the TF-IDF value for each term.
    """
    tf_idf_q = {}
    
    for word in query:
      tf_idf_q[word] =(1.0+math.log10(query.count(word))) * self.idf(word)

    return tf_idf_q

  def cosine_similarity(self,
                         query: List[str],
                         document: str) -> float:
    """Cosine Similarity function.

    Args:
      - query List[str]: A query. 
      - document (str): A document.

    Returns:
      - float : The value of the cosine similarity between the query and the document.
    """
    tf_idf_q = self.tf_idf_query(query)

    q_vect = list(tf_idf_q.values())
    doc_vect = [self.index_tf_idf[self.vocabulary.get_index(word)][self.binary_search(self.index_tf_idf[self.vocabulary.get_index(word)], document)][1] for word in query]

    length_doc = np.linalg.norm(doc_vect)
    length_q = np.linalg.norm(q_vect)

    num = np.dot(doc_vect, q_vect)
    den = length_doc*length_q

    return num/den

  def binary_search(self,
                   alist: Iterable[Tuple[str, Any]],
                   document: str) -> int:
    """An custom version of the binary search method.

    Args:
      - alist (Iterable[Tuple[str, Any]]): An Iterable of the format ("document_d", "tf_idf_d_w").
      - document (str): The document id we're searching.

    Returns:
      - pos (int): The position of the document in the alist.
    """
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
      pos = 0
      midpoint = (first + last)//2
      if alist[midpoint][0] == document:
          pos = midpoint
          found = True
      else:
          if int(document.split("_")[1]) < int(alist[midpoint][0].split("_")[1]):
              last = midpoint-1
          else:
              first = midpoint+1

    if not found:
      raise Exception(f"Document {document} not in word list.")

    return pos

  def load_tf_idf_index(self,
                        path: str) -> None:
    """The loader method used to load a saved Index.

    Args:
      - path (str): The path where the index to load is stored.

    Returns:
      - None
    """
    self.index_tf_idf = load(path)
    return None

  def clean_query(self,
                  query: str) -> List[str]:
    """We follow the same preprocessing steps we have done for the documents, but for the query.

    Args:
      -  query (str): The query.

    Returns:
      - List[str] : The cleaned version of the query.
    """
    query = query.lower()

    #We tokenize the query
    regexp = RegexpTokenizer('\w+')
    query = regexp.tokenize(query)

    # We remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    query = [item for item in query if item not in stopwords]

    # We execute stemming over the query
    porter_stemmer = PorterStemmer()
    query = [porter_stemmer.stem(item) for item in query]

    # Remove the words with less than 2 terms
    query = [item for item in query if len(item)>2]

    return query

  def query_index(self,
                  query) -> SortedSet:
    """A method to query the index.

    Args:
      - query (str): The Conjunctive query.

    Returns:
      - result_set (SortedSet): The result SortedSet of the query.
    """  
    for i, word in enumerate(query):
      if i == 0:
        result_set = self.index[self.vocabulary.get_index(word)]
      else:
        result_set = result_set.intersection(self.index[self.vocabulary.get_index(word)])

    return result_set

  def query_cosine_similarity(self,
                              query: str,
                              columns: List[str] = None,
                              k: int = None) -> List[Tuple[str, float]]:
    """Return the rows in the dataframe that answer to the query.

    Args:
      - query (str): The query to search for.
      - columns (List[str]): The dataframe columns to display, if none all of them. Default None.
      - k (int): Number of desired documents. Default None.

    Returns:
      - List[Tuple[int, str]] : A list of top-k documents with their ranking.
    """
    query = self.clean_query(query)

    if columns is None:
      columns = self.dataframe.columns

    copy = self.dataframe[columns].loc[self.query_index(query)].copy()
    copy["Similarity"] = copy.index.map(lambda x: self.cosine_similarity(query, str(x)))

    if k == None or k > len(copy):
      k = len(copy)

    sorted_copy = copy.sort_values(by="Similarity", ascending=False).head(k)

    display(sorted_copy)

    return [(i+1, doc) for i, doc in enumerate(sorted_copy.index)]

  def query(self,
            query: str,
            columns: List[str] = None) -> pd.DataFrame:
    """Return the rows in the dataframe that answer to the query.

    Args:
      - query (str): The query to search for.
      - columns (List[str]): The dataframe columns to display, if none all of them. Default None.

    Returns:
      - pd.DataFrame : The sub set of the dataframe that answers to the query.
    """
    query = self.clean_query(query)

    if columns is None:
      columns = self.dataframe.columns

    return self.dataframe[columns].loc[self.query_index(query)].sort_index(ascending=False)


class OurSearchEngine():
  """A class that rappresent our Search Engine, here we find all the methods needed to query on the dataframe.

  Args:
    - dataframe (pd.DataFrame): The pd.DataFrame to query.
    - vocabulary (Vocabulary): The vocabulary of the dataframe.
    - index: The index word->documents of the dataframe.

  Attributes:
    - dataframe (pd.DataFrame): The variable in which we save the dataframe input.
    - vocabulary (Vocabulary): The variable in which we save the vocabulary input.
    - index (Index): The variable in which we save the index input. 
  """
  def __init__(self,
               dataframe: pd.DataFrame,
               vocabulary: Vocabulary,
               index: Index):

    print("Sanity Check between the Vocabulary and the Index:", end="\t")
    if not len(vocabulary) == len(index) :
      raise Exception(f"Inconsistency between the number of words in the vocabulary and the number of words in the index.\nVocabulary: {len(vocabulary)}\nIndex: {len(index)}")

    print("[PASSED]")

    self.index = index
    self.dataframe = dataframe
    self.vocabulary = vocabulary

  def words(self) -> List[str]:
    """Returns the list of the words.

    Returns:
      - List[tr]: The list of the words.
    """
    return self.vocabulary.words()

  def tf(self,
         word: str,
         document: str,
         column: str) -> float:
    """The function that calculates TF of a word and a document.

    Args:
      - word (str): The word for which we calculate the TF.
      - document (str): The document for which we calculate the TF.
      - column (str): The dataframe column on which we calculate the TF. 

    Returns:
      - float : The TF value for that word and document.
    """
    frq_t_in_d = self.vocabulary.dataframe[column][document].count(word)

    if frq_t_in_d != 0:
      frq_t_in_d = 1.0 + math.log10(frq_t_in_d)
      
    return frq_t_in_d

  def idf(self,
          word: str) -> float:
    """The function that calculates IDF of a word and a document.

    Args:
      - word (str): The word for which we calculate the IDF.

    Returns:
      - float : The IDF value for that.
    """
    n_docs_with_word = len(self.index[self.vocabulary.get_index(word)])

    return math.log10((1.0+len(self.vocabulary.dataframe))/(1.0 + n_docs_with_word))+1.0

  def tf_idf(self,
             word: str,
             document: str,
             column: str) -> float:
    """The function that calculates TF-IDF of a word and a document.

    Args:
      - word (str): The word for which we calculate the TF-IDF.
      - document (str): The document for which we calculate the TF-IDF.
      - column (str): The dataframe column on which we calculate the TF. 

    Returns:
      - float : The TF value for that word and document.
    """
    return self.tf(word, document, column)*self.idf(word)

  def create_index_tf_idf(self,
                          columns: List[str] = None,
                          path: str = None) -> Dict[str, List[Tuple[str, float]]]:
    """A function that returns the index tf-idf on the corpora of the self.dataframe.

    Args:
      - path (str): The name of the pickle file where we save the index, if None the index is not saved. Default None.
      - columns (List[str]): The columns on which we create the index_tf_idf. If None we use all the dataframe columns. Default None. 

    Returns:
      - index_tf_idf (Dict[str, List[Tuple[str, float]]]): The index with the tf-idf.
    """
    if columns is None:
      columns = self.vocabulary.dataframe.columns

    print(f"The index is of the type:\nTerm_id -> [(document_id, {list(columns)})]")

    self.index_tf_idf = {}
    for word in tqdm(self.words()):
      self.index_tf_idf[self.vocabulary.get_index(word)] = []
      for doc in self.index[self.vocabulary.get_index(word)]:
        self.index_tf_idf[self.vocabulary.get_index(word)].append((doc, [self.tf_idf(word, doc, column) for column in columns]))

    if path:
      dump(self.index_tf_idf, path)

    return self.index_tf_idf

  def tf_idf_query(self,
                   query: List[str]) -> Dict[str, float]:
    """A function to calculate the TF-IDF for a query.

    Args:
      - query List[str]: A query.

    Returns
      - tf_idf_q (Dict[str, float]): The dictionary for the query with the TF-IDF value for each term.
    """
    tf_idf_q = {}
    
    for word in query:
      tf_idf_q[word] =(1.0+math.log10(query.count(word))) * self.idf(word)

    return tf_idf_q

  def cosine_similarity(self,
                        query: List[str],
                        document: str,
                        k: int) -> float:
    """Cosine Similarity function.

    Args:
      - query List[str]: A query.
      - document (str): A document.
      - k (int): The element on which we calculate the cosine similarity.

    Returns:
      - float : The value of the cosine similarity between the query and the document.
    """
    tf_idf_q = self.tf_idf_query(query)

    q_vect = list(tf_idf_q.values())
    doc_vect = [self.index_tf_idf[self.vocabulary.get_index(word)][self.binary_search(self.index_tf_idf[self.vocabulary.get_index(word)], document)][1][k] for word in query]

    length_doc = np.linalg.norm(doc_vect)
    length_q = np.linalg.norm(q_vect)

    num = np.dot(doc_vect, q_vect)
    den = length_doc*length_q

    if den == 0.0:
      result = 0.0
    else:
      result = num/den

    return result

  def binary_search(self,
                   alist: Iterable[Tuple[str, Any]],
                   document: str) -> int:
    """An custom version of the binary search method.

    Args:
      - alist (Iterable[Tuple[str, Any]]): An Iterable of the format ("document_d", "tf_idf_d_w").
      - document (str): The document id we're searching.

    Returns:
      - pos (int): The position of the document in the alist.
    """
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
      pos = 0
      midpoint = (first + last)//2
      if alist[midpoint][0] == document:
          pos = midpoint
          found = True
      else:
          if int(document.split("_")[1]) < int(alist[midpoint][0].split("_")[1]):
              last = midpoint-1
          else:
              first = midpoint+1

    if not found:
      raise Exception(f"Document {document} not in word list.")

    return pos

  def load_tf_idf_index(self,
                        path: str) -> None:
    """The loader method used to load a saved Index.

    Args:
      - path (str): The path where the index to load is stored.

    Returns:
      - None
    """
    self.index_tf_idf = load(path)
    return None

  def clean_query(self,
                  query: str) -> List[str]:
    """We follow the same preprocessing steps we have done for the documents, but for the query.

    Args:
      -  query (str): The query.

    Returns:
      - List[str] : The cleaned version of the query.
    """
    query = query.lower()

    #We tokenize the query
    regexp = RegexpTokenizer('\w+')
    query = regexp.tokenize(query)

    # We remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    query = [item for item in query if item not in stopwords]

    # We execute stemming over the query
    porter_stemmer = PorterStemmer()
    query = [porter_stemmer.stem(item) for item in query]

    # Remove the words with less than 2 terms
    query = [item for item in query if len(item)>2]

    return query

  def query_index(self,
                  query) -> SortedSet:
    """A method to query the index.

    Args:
      - query (str): The Conjunctive query.

    Returns:
      - result_set (SortedSet): The result SortedSet of the query.
    """  
    for i, word in enumerate(query):
      if i == 0:
        result_set = self.index[self.vocabulary.get_index(word)]
      else:
        result_set = result_set.intersection(self.index[self.vocabulary.get_index(word)])

    return result_set

  def our_score(self,
                query: List[str],
                document: str) -> float:
    """Cosine Similarity function.

    Args:
      - query List[str]: A query.
      - document (str): A document.

    Returns:
      - float : The value of our score between the query and the document.
    """
    return self.cosine_similarity(query, document, 2)*((1+self.cosine_similarity(query, document, 0)+self.cosine_similarity(query, document, 1)+self.cosine_similarity(query, document, 3))/4)

  def query_with_our_similarity(self,
                                query: str,
                                columns: List[str] = None,
                                k: int = None) -> List[Tuple[str, float]]:
    """Return the rows in the dataframe that answer to the query.

    Args:
      - query (str): The query to search for.
      - columns (List[str]): The dataframe columns to display, if none all of them. Default None.
      - k (int): Number of desired documents. Default None.

    Returns:
      - List[Tuple[int, str]] : A list of top-k documents with their ranking.
    """
    query = self.clean_query(query)

    if columns is None:
      columns = self.dataframe.columns

    copy = self.dataframe[columns].loc[self.query_index(query)].copy()
    copy["Similarity"] = copy.index.map(lambda x: self.our_score(query, str(x)))

    if k == None or k > len(copy):
      k = len(copy)

    sorted_copy = copy.sort_values(by="Similarity", ascending=False).head(k)

    display(sorted_copy)

    return [(i+1, doc) for i, doc in enumerate(sorted_copy.index)]

  def query(self,
            query: str,
            columns: List[str] = None) -> pd.DataFrame:
    """Return the rows in the dataframe that answer to the query.

    Args:
      - query (str): The query to search for.
      - columns (List[str]): The dataframe columns to display, if none all of them. Default None.

    Returns:
      - pd.DataFrame : The sub set of the dataframe that answers to the query.
    """
    query = self.clean_query(query)

    if columns is None:
      columns = self.dataframe.columns

    return self.dataframe[columns].loc[self.query_index(query)].sort_index(ascending=False)

class HeapDocs():
  """An heap data structure to mantain our Top K docs.

  Args:
    - data (List[Tuple[int, str]]): Our list of (ranking, doc_id).
  
  Attributes:
    - data (List[Tuple[int, str]]): The variable where we store the data input.
  """
  def __init__(self,
               data: List[Tuple[int, str]]):
    self.data = data
    heapq.heapify(data)

  def __len__(self) -> int:
    """An override of the '__len__()' method.

    Returns:
      - int : The length of our heap.
    """
    return len(self.data)

  def __str__(self) -> str:
    """An override of the '__str__()' method that is going to be used by print().

    Returns:
      - str : A string rappresentation of our heap
    """
    alist = [k for i, k in self.data]
    alist = '", "'.join(alist)
    return f'["{alist}"]'

  def pop(self) -> str:
    """Pop the biggest element from our heap.

    Returns:
      - str : The document id with the highest similarity score in the heap.
    """
    return heapq.heappop(self.data)[1]

  def top(self) -> str:
    """Return the biggest element from our heap.

    Returns
      - str : The document id with the highest similarity score in the heap.
    """
    return self.data[0][1]
