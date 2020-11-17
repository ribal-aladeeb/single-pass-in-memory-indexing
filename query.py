import argparse
import os
from shutil import Error
import nltk
from collections import Counter
import math
from typing import Dict
import utils
import indexer


QUERY_TYPES = ['RANKED', 'AND', 'OR']
L_AVG_FILENAME = 'average_doc_length.txt'
L_AVG_FILENAME = os.path.join(indexer.CACHING_DIR, L_AVG_FILENAME)


def init_params():
    parser = argparse.ArgumentParser(description='Process command line arguments')
    # parser.add_argument('query_terms', type=str, nargs='+', help='The query string used for your search')
    # parser.add_argument('-t', '--query_type', default='RANKED',
    #                     help='The type of query you want to run (AND finds the documents that contain all the query words. OR returns the documents that have at least 1 of the query terms')

    args = parser.parse_args()

    return args


def compute_ranking_RSV_11_32(terms, docID, L_avg, inverted_index, k=5, b=0.5, N=21578) -> float:
    '''
    This func computes the document ranking function 11.32 of the Stanford's NLP
    Information Retrieval textbook. Given a list of terms (i.e a query) and a
    docID, this func will return the RSV (i.e. the ranking) of a the document in
    reference to the term.

    Textbook online link:
    https://nlp.stanford.edu/IR-book/information-retrieval-book.html

    Chapter where you will finding the function (math and explanation) that this
    function attempts to implement:
    https://nlp.stanford.edu/IR-book/pdf/11prob.pdf (go to 11.32)
    '''

    L_d: int     # number of unique terms in document

    with open(os.path.join(indexer.DOCUMENT_DIR, str(docID)), mode='r') as f:
        document = f.read()
        tokens: list = nltk.RegexpTokenizer(indexer.TOKENIZING_REGEX).tokenize(document)
        terms: set = set(tokens)
        term_count = Counter(tokens)
        L_d = len(terms)

    RSV = 0
    for term in terms:
        tf: int = term_count[term]                           # term frequency in this specific document
        df: int = inverted_index.get(term, (0, []))[0]       # document frequency of this specific term (length of its postings list)

        RSV += math.log10(N/df) * (k+1)*tf / (k*((1-b) + b*(L_d/L_avg)) + tf)

    return RSV


def intersect_postings(terms, inverted_index) -> set:
    '''
    Returns the intersection of the postings list of the terms list. Used for
    AND queries.
    '''

    assert len(terms) > 0, 'terms cannot be empty'

    _, intersected = inverted_index.get(terms[0], (0, []))

    for i in range(1, len(terms)):
        _, postings = inverted_index.get(terms[i], (0, []))
        postings = set(postings)
        intersected.intersection(postings)

    return intersected


def union_postings(terms, inverted_index) -> Dict[int, str]:
    '''
    Returns the intersection or union of the postings list of the terms list
    depending on whether the query_type is AND or OR respectively. Instead of a
    set, it returns a dictionary of the union where the docIDs are keys and the
    number of query terms that appear in said document.
    '''

    assert len(terms) > 0, 'terms cannot be empty'

    unioned = {}

    for term in terms:
        _, postings = inverted_index.get(term, (0, []))

        postings = set(postings)
        for docID in postings:
            unioned[docID] = unioned.get(docID, 0) + 1

    return unioned


def find_average_document_length() -> int:
    '''
    Tries to read contents of L_AVG_FILENAME. If it doesn't exist, it will open
    each document in indexer.DOCUMENT_DIR and compute the average length of
    documents. If indexer.DOCUMENT_DIR does not exist it will create it.
    '''

    if os.path.isfile(L_AVG_FILENAME):
        with open(L_AVG_FILENAME, mode='r') as f:
            try:
                L_avg = float(f.read())
                return L_avg

            except Exception as e:  # The file exist but not castable to float
                print(e)
                os.remove(L_AVG_FILENAME)

    if not os.path.isdir(indexer.DOCUMENT_DIR):
        indexer.save_extracted_docs(indexer.extract_documents())

    document_files = os.listdir(indexer.DOCUMENT_DIR)
    doc_lengths = []
    for filename in document_files:
        with open(os.path.join(indexer.DOCUMENT_DIR, filename), mode='r') as f:
            contents = f.read()
            tokens = nltk.RegexpTokenizer(indexer.TOKENIZING_REGEX).tokenize(contents)
            terms = set(tokens)
            length = len(terms)
        doc_lengths.append(length)

    L_avg = sum(doc_lengths)/len(doc_lengths)

    with open(L_AVG_FILENAME, mode='w') as f:
        f.write(str(L_avg))

    return L_avg


def main():

    args = init_params()
    inverted_index = utils.load_index(indexer.INVERTED_INDEX_FILE)

    query_type = {
        'a': 'AND',
        'and': 'AND',
        'o': 'OR',
        'or': 'OR',
        'r': 'RANKED',
        'ranked': 'RANKED',
    }
    requested = ''
    while requested not in query_type:
        if requested != '':
            print(f'\n"{requested}" is not a valid type of search.')
        requested = input("Enter the type of search you want to perform ([a]nd, [o]r, [r]anked): ").lower()

    results = ''
    if query_type[requested] == 'RANKED':
        query_terms = input("Please enter your query: ").split(' ')
        postings = list(union_postings(query_terms, inverted_index).keys())
        ranking = {}
        L_avg = find_average_document_length()
        for docID in postings:
            ranking[docID] = round(compute_ranking_RSV_11_32(query_terms, docID, L_avg, inverted_index), 2)

        sorted_rankings = sorted(ranking.items(), reverse=True, key=lambda x: x[1])
        top10 = sorted_rankings[:10]

        if len(top10) == 0:
            results = '\nNo results found, sorry!'

        for i in range(len(top10)):
            results += f'\n{i+1}. \tDocument ID {top10[i][0]} \twith ranking {top10[i][1]}'

    else:
        print('ta mere')

    print(results)


if __name__ == '__main__':
    main()
