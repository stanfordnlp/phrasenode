#!/usr/bin/env bash

PROCESSED_PAGES=https://nlp.stanford.edu/projects/phrasenode/processed-pages.zip
DATASET=https://nlp.stanford.edu/projects/phrasenode/dataset-final.zip
PROCESSED_VECTORS=https://nlp.stanford.edu/projects/phrasenode/processed-glove.zip
VOCAB=https://nlp.stanford.edu/projects/phrasenode/vocab.zip

PROCESSED_PAGES_PATH=data/phrase-node-dataset/infos
DATASET_PATH=data/phrase-node-dataset/data
PROCESSED_VECTORS_PATH=data/glove
VOCAB_PATH=data/vocab

echo "make data directory."
mkdir -p $PROCESSED_PAGES_PATH
mkdir -p $DATASET_PATH
mkdir -p $PROCESSED_VECTORS_PATH
mkdir -p $VOCAB_PATH

echo "dataset downloading and unpacking..."
wget -P $PROCESSED_PAGES_PATH $PROCESSED_PAGES
unzip $PROCESSED_PAGES_PATH/processed-pages.zip -d $PROCESSED_PAGES_PATH
wget -P $DATASET_PATH $DATASET
unzip $DATASET_PATH/dataset-final.zip -d $DATASET_PATH
wget -P $PROCESSED_VECTORS_PATH $PROCESSED_VECTORS
unzip $PROCESSED_VECTORS_PATH/processed-glove.zip -d $PROCESSED_VECTORS_PATH
wget -P $VOCAB_PATH $VOCAB
unzip $VOCAB_PATH/vocab.zip -d $VOCAB_PATH

echo "dataset downloading and unpacking completed!"