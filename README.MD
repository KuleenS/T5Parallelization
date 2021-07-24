# Parallelize T5
## Explanation
### Data
Txt files consisting of the raw training data 
### modeldump 
Premade model storing area
### environ
Conda environment for this program to run\
environment's name is nlp\
This was on CUDA 10.2\
conda env create --file environ.yml
### ParallelizeT5.py
The Main Script
## Using
ParallelizeT5.py has 3 arguments: \
Batch Size: -b int\
Epochs: -e int\
Number of GPUs: -g int\
Debug: -d int (0 or 1))\
I would recommend 10-15 epochs \
The batch size was around 8-12 on my machine but you can play around with it \
**The debug feature is very important as you can quickly run through the script in order to play around with the batch size**
