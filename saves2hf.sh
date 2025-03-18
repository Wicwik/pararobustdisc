#!/bin/bash

for i in `ls -d saves/*best`; 
do
    python 2hf.py $i
done
