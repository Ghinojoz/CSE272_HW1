import lucene
lucene.initVM()
import os
import xml.etree.ElementTree as ET
import time
import sys

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexReader
from org.apache.lucene.index import Term
from org.apache.lucene.document import Document
from org.apache.lucene.document import Field
from org.apache.lucene.document import StringField
from org.apache.lucene.document import TextField
from org.apache.lucene.store import Directory
from org.apache.lucene.store import ByteBuffersDirectory
from org.apache.lucene.analysis.en import EnglishAnalyzer

from org.apache.lucene.queryparser.classic import ParseException
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser

from org.apache.lucene.search import IndexSearcher
#from org.apache.lucene.search import Query
#from org.apache.lucene.search import ScoreDoc
#from org.apache.lucene.search import TopDocs

from org.apache.lucene.search.similarities import BooleanSimilarity
from org.apache.lucene.search.similarities import TFIDFSimilarity

from org.apache.lucene.search import BooleanQuery, BooleanClause, TermQuery, Explanation, Query, ScoreDoc, TopDocs
# from org.apache.lucene.search import BooleanClause
# from org.apache.lucene. search import TermQuery

# for producing custom similarity classes
from org.apache.pylucene.search import PythonSimpleCollector
from org.apache.pylucene.search.similarities import PythonClassicSimilarity
import math

class My_TFSimilarity(PythonClassicSimilarity):
    def lengthNorm(self, numTerms):
        return 1.0

    def tf(self, freq):
        return math.sqrt(freq)

    def sloppyFreq(self, distance):
        return 2.0

    def idf(self, docFreq, numDocs):
        return 1.0

    def idfExplain(self, collectionStats, termStats):
        return Explanation.match(1.0, 'inexplicable', [])

class My_TFIDFSimilarity(PythonClassicSimilarity):
    def lengthNorm(self, numTerms):
        return 1.0

    def tf(self, freq):
        return math.sqrt(freq)

    def sloppyFreq(self, distance):
        return 2.0

    def idf(self, docFreq, numDocs):
        return math.log((numDocs/(float(docFreq) + 1.0)), 10.0) + 1.0

    def idfExplain(self, collectionStats, termStats):
        return Explanation.match(1.0, 'inexplicable', [])


def create_inverted_index(f_name, index, analyzer, my_similarity):
    config = IndexWriterConfig(analyzer)
    config.setSimilarity(my_similarity)
    writer = IndexWriter(index, config)

    f = open(f_name, 'r')

    line = f.readline()
    count = 0
    while line:
        # create a new doc for the index
        doc = Document()

        # parse this index line for this doc
        doc_ind = line.split(' ')
        doc.add(StringField('sequential_identifier', doc_ind[1], Field.Store.YES))
        count += 1
        line = f.readline()
        everything_field = ''
        while line and not line.startswith('.I'):
            # read lines and populate as much information for this document as we can

            if line.startswith('.U'):
                line = f.readline()
                line = line.strip('\n')
                # this is what is used for trec-eval identification
                doc.add(StringField('medline_identifier', line, Field.Store.YES))

            if line.startswith('.M'):
                line = f.readline()
                line = line.strip('\n')
                doc.add(TextField('mesh_terms', line, Field.Store.YES))

            if line.startswith('.T'):
                line = f.readline()
                line = line.strip('\n')
                everything_field += line
                doc.add(TextField('title', line, Field.Store.YES))

            if line.startswith('.P'):
                line = f.readline()
                line = line.strip('\n')
                doc.add(TextField('publication_type', line, Field.Store.YES))

            if line.startswith('.W'):
                line = f.readline()
                line = line.strip('\n')
                everything_field += line
                doc.add(TextField('abstract', line, Field.Store.YES))

            if line.startswith('.A'):
                line = f.readline()
                line = line.strip('\n')
                doc.add(TextField('authors', line, Field.Store.YES))

            if line.startswith('.S'):
                line = f.readline()
                line = line.strip('\n')
                doc.add(TextField('source', line, Field.Store.YES))

            line = f.readline()

        # append the document and proceed
        # this is for combining title and abstract if one exists
        doc.add(TextField('everything', everything_field, Field.Store.YES))
        writer.addDocument(doc)

    f.close()
    writer.close()

    print('Number according to count: ', count)
    return index


# returns a list of queries, which is a tuple containing
# (query number, title, description)
def parse_queries(f_name):
    queries = []

    f = open(f_name, 'r')
    line = f.readline()

    while line:
        if not line.startswith('<top>'):
            line = f.readline()
        else:
            query_number = None
            query_title = None
            query_description = None

            # get the query number
            line = f.readline()
            num_split = line.split(': ')
            query_number = num_split[1].strip('\n')
            query_number = query_number.replace('/', ' ')

            # get the query title
            line = f.readline()
            title_split = line.split('> ')
            query_title = title_split[1].strip('\n')
            query_title = query_title.replace('/', ' ')

            # get the query description
            line = f.readline()
            line = f.readline()
            query_description = line.strip('\n')
            query_description = query_description.replace('/', ' ')

            queries.append((query_number, query_title, query_description))
            line = f.readline()

    return queries


# (query number, title, description)
def make_boolean_queries(all_queries, searcher, analyzer):
    query_result_pairs = []
    for current_query in all_queries:
        my_query = BooleanQuery.Builder()
        my_query_parser = QueryParser('everything', analyzer)
        q = my_query_parser.parse(current_query[2])
        q_string = q.toString()
        q_tokens = q_string.split(' ')
        for token_pair in q_tokens:
            field_token_pair = token_pair.split(':')
            my_field = field_token_pair[0]
            my_token = field_token_pair[1]
            my_term = Term(my_field, my_token)
            my_term_query = TermQuery(my_term)
            my_query.add(my_term_query, BooleanClause.Occur.MUST)
        bool_query = my_query.build()
        docs = searcher.search(bool_query, 50)

        query_result_pairs.append((current_query[0], docs.scoreDocs))
    return query_result_pairs


def make_non_boolean_queries(all_queries, searcher, analyzer):
    query_result_pairs = []
    for current_query in all_queries:
        my_query = QueryParser('everything', analyzer).parse(current_query[2])
        docs = searcher.search(my_query, 50)
        query_result_pairs.append((current_query[0], docs.scoreDocs))
    return query_result_pairs

if __name__ == '__main__':
    query_type = None
    if len(sys.argv) > 1:
        # we have been given a parameter defining the type of query
        query_type = sys.argv[1]
    else:
        query_type = 'boolean'

    # select similarity function based on type of query we are doing
    my_similarity = None
    if query_type == 'boolean':
        my_similarity = BooleanSimilarity()
    elif query_type == 'tf':
        print('TF SIMILARITY!')
        my_similarity = My_TFSimilarity()
    elif query_type == 'tfidf':
        print('TFIDF SIMILARITY!')
        my_similarity = My_TFIDFSimilarity()
    elif query_type == 'custom':
        # update later
        my_similarity = TFIDFSimilarity()

    # sanity check, make sure we have something for similarity
    if my_similarity is None:
        print("Error, no similarity initialized")
        exit()

    print(EnglishAnalyzer.getDefaultStopSet())
    analyzer = StandardAnalyzer(EnglishAnalyzer.getDefaultStopSet())
    # print('Analyzer stop words: ', len(analyzer.getStopwordSet()))
    index = ByteBuffersDirectory()
    # my_similarity = BooleanSimilarity()

    index = create_inverted_index('ohsumed.88-91', index, analyzer, my_similarity)

    # determine which kind of query we are doing, and record the name for the log file
    # default to doing boolean query

    log_file = query_type + '.log'

    reader = DirectoryReader.open(index)
    print('Num Documents:', reader.numDocs())
    print('Query Type: ', log_file)

    # get the information for the queries
    all_queries = parse_queries('query.ohsu.1-63')

    #q = QueryParser('title', analyzer)

    searcher = IndexSearcher(reader)
    searcher.setSimilarity(my_similarity)

    # get results based on type of search we are doing
    results = None

    if query_type == 'boolean':
        results = make_boolean_queries(all_queries, searcher, analyzer)
    else:
        results = make_non_boolean_queries(all_queries, searcher, analyzer)

    if results is None:
        print('Error: Results is empty!')
        exit()
    log = open(log_file, 'w')
    for result in results:

        rank = 1
        for ranked_doc in result[1]:
            log.write( ' '.join([str(result[0]), 'Q0', str(reader.document(ranked_doc.doc).get('medline_identifier')), str(rank), str(ranked_doc.score), query_type, '\n']))
            # print(result[0], ' Q0 ', reader.document(ranked_doc.doc).get('medline_identifier'), ' ', rank, ' ', ranked_doc.score, ' boolean')
            rank += 1

    # create the searcher

    reader.close()
    log.close()
