import indexer

if __name__ == '__main__':
    docs: dict = indexer.document_extracter(indexer.unpack_corpus_step1(indexer.DATA_DIR))
    indexer.generate_and_save_blocks_to_disk(docs)
