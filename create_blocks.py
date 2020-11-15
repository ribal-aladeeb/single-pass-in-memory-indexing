import indexer
import time


if __name__ == '__main__':
    start_time = time.time()

    docs: dict =indexer.document_extracter(indexer.unpack_corpus_step1(indexer.DATA_DIR))
    indexer.generate_and_save_blocks_to_disk(docs)

    elapsed = round(time.time() - start_time, 2)
    print(f'\nBlocks created in {elapsed} seconds')
