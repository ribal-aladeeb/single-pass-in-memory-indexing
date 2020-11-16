import os
from typing import Dict, Tuple, List, Iterable
import json
from tqdm import tqdm


def ensure_dir_exists(dir):
    '''Checks that <path> exists, otherwise creates it'''

    if os.path.isdir(dir):
        return

    os.mkdir(dir)
    print(f'Created {dir} directory')


def save_index_to_disk(inverted_index: Dict[str, Tuple[int, List[int]]], outfile: str) -> None:
    assert type(outfile) == str, "When using function save_index_to_disk you have to provide a string outfile arg."

    printable: List[Tuple[str, Tuple[int, List[int]]]] = sorted(inverted_index.items(), key=lambda token: token[0])
    write2disk(printable, outfile)


def write2disk(lines: Iterable, outfile) -> None:
    '''
    This is a general utility for writing intermediate and final computations
    into output files. The <lines> argument needs to be iterable.
    '''
    assert outfile != None, "when using function write2disk, outfile arg cannot be None"

    if type(outfile) == str:
        with open(outfile, mode='w', encoding='UTF-8') as f:
            for l in lines:
                print(json.dumps(l), file=f)

    else:  # output must be sys.stdout (so it's already a file obj)
        for l in lines:
            print(json.dumps(l), file=outfile)


def load_json_from_disk(infile):
    '''Reads json obj from infile.'''

    if type(infile) == str:
        with open(infile, mode='r', encoding='UTF-8') as f:
            obj = json.load(f)

    else:  # infile must be a file obj
        obj = json.load(infile)

    return obj

def load_index(filename: str) -> Dict[str, Tuple[int, List[int]]]:
    '''Loads the inverted index stored in <filename>.'''

    # print(f"\nLoading inverted index from {filename}")

    inv_idx = []                                           # as loaded by json.loads(), type(inv_idx) == List[str, List[int,List[int]]]
    inverted_index: Dict[str, Tuple[int, List[int]]] = {}  # This will contain the correct format and type

    with open(filename, mode='r') as file:
        i = 0
        for line in file:
            inv_idx.append(json.loads(line))
            token: str = inv_idx[i][0]
            frequency: int = inv_idx[i][1][0]
            postings_list: List[int] = inv_idx[i][1][1]
            inverted_index[token] = (frequency, postings_list)
            i -= -1

    return inverted_index
