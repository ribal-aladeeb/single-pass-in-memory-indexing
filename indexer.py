import argparse
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
import nltk
import utils
import time
import shutil

CACHING_DIR = 'cache/'
BLOCK_DIR = 'blocks/'
BLOCK_LENGTH = 500
DATA_DIR = 'data/'
DOCUMENT_DIR = 'documents/'
INVERTED_INDEX_FILE = 'inverted_index.txt'
TOKENIZING_REGEX = r"[a-zA-Z]+[-']{0,1}[a-zA-Z]*[']{0,1}"

BLOCK_DIR = os.path.join(CACHING_DIR, BLOCK_DIR)
DOCUMENT_DIR = os.path.join(CACHING_DIR, DOCUMENT_DIR)
INVERTED_INDEX_FILE = os.path.join(CACHING_DIR, INVERTED_INDEX_FILE)


def init_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--block_size', type=int, default=10000, help="Number of terms in each block")
    return parser.parse_args()


def unpack_corpus_step1(path: str) -> List[str]:
    '''Given corpus <path>, returns a list where element is the full str content of a file'''

    sgmfiles: List[str] = sorted([fname for fname in os.listdir(path) if '.sgm' in fname])
    files_contents = []

    for filename in sgmfiles:
        with open(f'{path}/{filename}', mode='r', encoding='UTF-8', errors='ignore') as file:
            files_contents += [file.read()]

    return files_contents


def document_extracter(lines: List[str]) -> dict:
    '''
    Returns a dictionary of doc_id -> doc_contents.
    <lines> is the contents of each .sgm file from the <unpack_corpus_step1>
    func.
    It is a design decision to include only the contents of the text tags
    for the construction of the index.
    '''

    print("Extracting documents from .sgm file contents")
    docs = {}
    for line in tqdm(lines):
        still_looking = True
        start_idx = end_idx = 0

        while still_looking:
            start_idx = line.find('<REUTERS', start_idx)

            if start_idx >= 0:  # means that the file still contains the substring

                docID = extract_docid(line, start_idx)

                end_idx = line.find('</REUTERS>', start_idx) + len('</REUTERS>')
                assert end_idx > 0, f"The end Index is negative {end_idx}"

                document: str = line[start_idx:end_idx]

                clean_doc: str = clean_reccuring_patterns(remove_tags(extract_text_contents(document)))
                docs[docID] = clean_doc

                start_idx = end_idx

            else:
                still_looking = False

    return docs


def extract_docid(sgmdoc: str, idx: int) -> int:
    '''
    Given a raw sgm file content string, extract the NEWID tag value of the
    reuters doc starting at index <idx>.
    '''
    id_start = sgmdoc.find('NEWID="', idx) + len('NEWID="')
    id_end = sgmdoc.find('"', id_start)

    try:
        docID = int(sgmdoc[id_start:id_end])

    except Exception as e:
        print("should not end up here")
        print(f"sgmdoc[id_start:id_end]=[{sgmdoc[id_start:id_end]}]")
        raise(e)

    return docID


def extract_text_contents(doc: str) -> List[str]:
    return doc[doc.find('<TEXT'):doc.find('</TEXT>')+len('</TEXT>')]


def remove_tags(doc: str) -> str:
    '''
    Removes any tags inside of doc. Should only be applied to a string that
    starts and ends with <TEXT> </TEXT> (output of extract_text_tag_contents
    func).
    '''
    doc_content = doc
    tag_start = doc_content.find('<')

    while tag_start >= 0:  # implies the substring exists

        tag_end: int = doc_content.find('>', tag_start)
        tag_type: str = doc_content[tag_start+1:tag_end]

        if tag_type != '/TEXT':
            '''
            Implications:
                1. Tag must be TITLE or BODY or DATE and needs to be
                removed.
                2. Add a space so that two words surrounding the tags
                don't get merged into a single one during tokenization purposes.
            '''
            doc_content = doc_content[:tag_start] + doc_content[tag_end+1:]

        else:  # must have reached end of document content
            doc_content = doc_content[:tag_start]

        tag_start = doc_content.find('<')

    return doc_content


def clean_reccuring_patterns(doc: str) -> str:
    '''
    ['&#2;', '&#3;'] are two patterns that occur very often at the begining and
    end of documents respectively. Filter them out of documents before create
    the term-docID pairs reduces the number of pairs from 4.5M to 3M pairs and
    from 17 to 12 seconds of processing.
    '''
    for pattern in ['&#2;', '&#3;']:
        start = doc.find(pattern)
        end = start + len(pattern) + 1
        doc = doc[:start] + ' ' + doc[end:]

    return doc


def save_extracted_docs(docs: Dict[int, str], dir=DOCUMENT_DIR) -> bool:
    '''
    This func will save the extracted contents into distinct files so that if
    you want to see the contents of document 7896, you don't need to parse the
    whole sgm files to look for the one containing it and to extract the reuters
    tag containing it. Instead you can just open the ./documents/7896 text file.
    This will be useful later to return the relevant query results.
    '''

    utils.ensure_dir_exists(dir)

    print(f"Writing the extracted docs into the {dir} folder")

    for ID in docs:
        with open(os.path.join(dir, str(ID)), mode='w') as f:
            f.write(docs[ID])

    return True


def load_extracted_docs(dir=DOCUMENT_DIR) -> Dict[int, str]:
    '''
    This function will load docs from the documents/ folder and return the exact
    same format as the document_extracter function.
    '''

    if not os.path.isdir(dir):
        return {}

    docs = {}

    for filename in os.listdir(dir):
        with open(os.path.join(dir, filename), mode='r') as f:
            ID = int(filename)
            docs[ID] = f.read()

    return docs


def extract_documents(data_dir=DATA_DIR, doc_dir=DOCUMENT_DIR, regex=TOKENIZING_REGEX):
    '''
    This function will extract the documents into a dictionary mapping of ID ->
    cleaned string contents. If the documents have previously been extracted in
    the DOCUMENT_DIR folder, it will take the shortcut of reading those file.
    Otherwise it will go over the sgm corpus files.
    '''

    docs: dict = load_extracted_docs()
    if len(docs) == 0:  # the documents have not previously been extracted into the DOCUMENT_DIR
        docs = document_extracter(unpack_corpus_step1(DATA_DIR))

    return docs


def generate_and_save_blocks_to_disk(docs: dict, dir=BLOCK_DIR, regex=TOKENIZING_REGEX, K=BLOCK_LENGTH) -> None:
    '''
    Creates the intermediate K-term dictionaries and stores them in BLOCK_DIR
    in files block_x.txt where x is monotonically increasing.
    '''

    def _save_block(block, block_count):
        for token in block:
            freq: int
            postings: set
            freq, postings = block[token]
            block[token] = freq, sorted(list(postings))

        filename = os.path.join(dir, f'block_{block_count}.txt')
        utils.save_index_to_disk(block, filename)

    print("Generating and saving block indices")

    if os.path.isdir(dir):
        shutil.rmtree(dir)

    utils.ensure_dir_exists(dir)

    block = {}
    block_count = 1

    for docID, contents in tqdm(sorted(docs.items(), key=lambda x: x[0])):
        tokenizer = nltk.RegexpTokenizer(regex)
        tokens = tokenizer.tokenize(contents)

        for token in tokens:
            frequency, postings = block.get(token, (0, set()))
            postings.add(docID)
            frequency = len(postings)
            block[token] = (frequency, postings)

            if len(block) == K:
                _save_block(block, block_count)
                block = {}
                block_count += 1

    print(f'Blocks are of length {K}')
    print(f'Length of last block {len(block)}')
    _save_block(block, block_count)  # This is the last block that does not get filled all the way, still need to save it
    return


def merge_blocks_into_one_index(dir=BLOCK_DIR) -> Dict[str, Tuple[int, List[int]]]:
    '''
    Merge all block indices in BLOCK_DIR into a single inverted index.
    '''

    inverted_index = {}
    print('Merging all blocks')
    for block_file in tqdm(os.listdir(dir)):
        block = utils.load_index(os.path.join(dir, block_file))
        for token in block:
            postings: set = inverted_index.get(token, set())
            _, block_postings = block[token]
            postings.update(block_postings)
            inverted_index[token] = postings

    for token in inverted_index:
        postings: set = inverted_index[token]
        postings: list = sorted(list(postings))
        freq = len(postings)

        inverted_index[token] = freq, postings

    return inverted_index


def main():
    block_size = init_params().block_size
    if block_size <= 0:
        print("Block size cannot be less than 1!")
        exit()
    BLOCK_LENGTH = block_size

    start_time = time.time()

    docs: dict = extract_documents()
    generate_and_save_blocks_to_disk(docs, K=BLOCK_LENGTH)
    inverted_index = merge_blocks_into_one_index()
    print('Blocks merged, saving complete index to disk')
    utils.save_index_to_disk(inverted_index, outfile=INVERTED_INDEX_FILE)

    elapsed = round(time.time() - start_time, 2)
    print(f'\nSPIMI performed in {elapsed} seconds')

    if not os.path.isdir(DOCUMENT_DIR):
        save_extracted_docs(docs)


if __name__ == '__main__':
    main()
