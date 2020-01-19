#!/bin/bash

mkdir -p data
wget https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/features_en_de.tar.gz -O data/
cd data/
tar zxvf features_en_de.tar.gz
mv en_de features
