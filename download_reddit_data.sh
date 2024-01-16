# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash

if [ ! -d "./data" ]; then
    mkdir ./data
fi

if [ ! -d "./data/raw_all" ]; then
    mkdir ./data/raw_all
fi

if [ ! -d "./data/iur_dataset" ]; then
    mkdir ./data/iur_dataset
fi

# Download data
curl https://storage.googleapis.com/naacl21_account_linking/raw_mud.tar.gz --output data/raw_all/1mil.tar.gz
curl https://storage.googleapis.com/naacl21_account_linking/raw_reddit_valtest.tar.gz --output data/raw_all/raw_reddit_valtest.tar.gz

# Untar data
tar zxvf data/raw_all/1mil.tar.gz -C data
tar zxvf data/raw_all/raw_reddit_valtest.tar.gz -C data/raw_all

# Create IUR dataset
cp data/raw_all/validation_queries.jsonl data/iur_dataset/train.jsonl
cp data/raw_all/validation_targets.jsonl data/iur_dataset/validation.jsonl
cp data/raw_all/test_queries.jsonl data/iur_dataset/test_queries.jsonl
cp data/raw_all/test_targets.jsonl data/iur_dataset/test_targets.jsonl

# Cleanup
rm data/raw_all/*.tar.gz
