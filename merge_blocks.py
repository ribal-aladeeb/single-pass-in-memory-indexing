import indexer
import utils

if __name__ == '__main__':
    inverted_index: dict = indexer.merge_blocks_into_one_index()
    utils.save_index_to_disk(inverted_index, outfile='inverted_index.txt')
