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
ParallelizeT5.py has 1 argument which is a json file describing how you want to train your T5.\
There are 12 options for you to change in the json file
1. Training type - unsupervised or supervised
2. Batch Size - how big the batches are
3. Epochs - how long you want to train your model
4. Num of GPUS - useful if you want to split training over mutliple gpus
5. Training Data Path
6. Output folder
7. Tokens to replace in your training data quickly
8. Tokens to add to your tokenizer 
9. If the model was previously trained, you can load in a new one
10. If you want to train the sample model multiple times - this allows for you to train the model unsupervised and then supervised
11. debug mode - if you want to run a smaller portion to test your batch sizes
12. size of model - how big do you want your T5 model 

Example structure:
{
     training: [{"unsupervised": parameters for training }, {"supervised": parameters for training }, {"supervised": parameters for training}],
     size: 3,
     debug: true
}

Remember that if you want to the same model with multiple training modes you must have train consec true until the last training mode which is false

**The debug feature is very important as you can quickly run through the script in order to play around with the batch size**
## Example
`python ParallelizeT5.py -j example_training.json` to run a model on the example json\
