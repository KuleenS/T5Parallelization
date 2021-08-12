<img src="https://huggingface.co/transformers/_images/parallelism-tp-parallel_shard_processing.png"
     alt="Parallelize"
     height = 200/>
# Parallelize T5
## Explanation
### Data
txt files consisting of the raw training data 
### Modeldump 
Premade model storing area
### Environment
conda environment for this program to run\
environment's name is nlp\
This was on CUDA 10.2\
`conda env create --file environ.yml`
## ParallelizeT5.py
The main script
## Using
ParallelizeT5.py has 3 arguments: \
Batch Size: -b: int\
Epochs: -e: int\
Number of GPUs -g: int\
Debug: -d int: (0 or 1)\
I would recommend 10-15 epochs \
The batch size was around 8-12 on my machine of 4 NVIDIA P100s but you can play around with it \
**The debug feature is very important as you can quickly run through the script in order to play around with the batch size**
## Example
`python ParallelizeT5.py -b 16 -g 4 -d 0` to run a model on 4 gpus, 16 batch size, and not on debug mode\
`python ParallelizeT5.py -b 16 -g 4 -d 1` to run the same as above but on debug mode
