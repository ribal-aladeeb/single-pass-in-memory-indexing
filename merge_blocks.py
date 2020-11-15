import indexer
import utils
import time


if __name__ == '__main__':
    start_time = time.time()

    inverted_index: dict = indexer.merge_blocks_into_one_index()
    utils.save_index_to_disk(inverted_index, outfile='inverted_index.txt')

    elapsed = round(time.time() - start_time, 2)
    print(f'\nBlocks merged in {elapsed} seconds')
