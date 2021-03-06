import torch
import os
import argparse
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5TokenizerFast
from datasets import Dataset
import pandas as pd
import json


#parses the json input
def parse_json(data):
    #check if the size of the model is too big or too small
    if data['size']>4 or data['size']<0 or not isinstance(data['size'], int):
        raise ValueError('Size Parameter is Invalid')
    
    #check if the debug value is invalid 
    if data['debug']<0 or data['debug']>1:
        raise ValueError('Debug Parameter is Invalid')
    
    #loop through the stages of training for T5
    for i in range(len(data['training'])):
        segment = list(data['training'][i].keys())[0]
        
        #if the segment is unsupervised add unsupervised segment
        if segment=="unsupervised":
            data_segment = data['training'][i]["unsupervised"]
        
        #if the segment is supervised add supervised segment
        elif segment=="supervised":
            data_segment = data['training'][i]["supervised"]
        replacement_tokens = data_segment["replacement_tokens"]
        
        #check if the replacement tokens is a file and its not blank string
        if not os.path.isfile(replacement_tokens) and replacement_tokens!="":
            raise ValueError(f'Replacement Token Files doesnt exist in {segment} {i+1}')
        
        #check if the training data is a file
        filepath = data_segment["data_path"]
        if not os.path.isdir(filepath):
            raise ValueError(f'Data File Path doesnt exist in {segment} {i+1}')
        
        #check if the extra tokens is a file and its not a blank string
        extra_tokens = data_segment['extra_tokens']
        if not os.path.isfile(extra_tokens) and extra_tokens!="":
            raise ValueError(f'Extra Token Files doesnt exist in {segment} {i+1}')
        
        #check if the previous trained model path exists
        if data_segment['previously_trained']:
            previously_trained_model = data_segment['previously_trained_model']
            if not os.path.isdir(previously_trained_model):
                raise ValueError(f'Previously Trained models doesnt exist in {segment} {i+1}')
         
        #checking if the consecutive booleans line up 
        if i==(len(data['training'])-1) and data_segment['train_consec']:
            raise ValueError(f'Wrong Train Consec parameter in {segment} {i+1}')


#returns the length of the vocab minus one
#used as an auxillary function for the tokenizing
def sentinel_id(vocab: int) -> int:
    return vocab-1

#returns input sentinel masked tokens using a combination of tensor tricks
#inputs tokens: tensor of integers, noise_mask: boolean tensor same size as tokens, vocab: integer that is length of tokenizer
def noise_span_to_unique_sentinel(tokens: torch.Tensor, noise_mask: torch.Tensor, vocab: int) -> torch.Tensor:
    prev_token_is_noise = torch.nn.functional.pad(noise_mask[:-1], ((1,0)))
    first_noise_tokens = torch.logical_and(noise_mask, torch.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)
    sentinel = sentinel_id(vocab) + 1 - torch.cumsum(first_noise_tokens.type(tokens.dtype), dim=0)
    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))

#returns the output sentinel masked tokens
def nonnoise_span_to_unique_sentinel(tokens: torch.Tensor, noise_mask: torch.Tensor, vocab:int) -> torch.Tensor:
    return noise_span_to_unique_sentinel(tokens, torch.logical_not(noise_mask), vocab)

def process_unsupervised_files(filepath: str, specialtokens: str):
    """
    filepath - a string to the path of where the files are being held
    NOTE: Make sure the files are in this format
        sentence sentence sentence\n
        sentence sentence sentence\n
        sentence sentence sentence\n
        ...
    specialtokens - a string to the path where the tokens you want removed are 
    NOTE: Make sure the file is in this format
        token\n
        token\n
        token\n
    """
    print('Starting Data loading and cleaning')
    #get list of files 
    files = os.listdir(filepath)
    target  = []
    
    #get the data from files into an array
    for i in range(len(files)):
        filename = os.path.join(filepath,files[i])
        with open(filename, "r") as f:
            target.extend(f.readlines())
    
    #replacing the special tokens in unsupervised files
    if specialtokens!="":
        replacement_tokens = open(specialtokens, "r").read().split("\n")
        for i in range(len(target)):
            for j in range(len(replacement_tokens)):
                target[i] = target[i].replace(replacement_tokens[j], '')
    
    print('Ending Data loading and cleaning')
    return target

def process_supervised_files(filepath: str, specialtokens: str):
    """
    filepath - a string to the path of where the files are being held
    NOTE: Make sure the files are csvs with this format
        sentence, sentence\n
        sentence, sentence\n
        sentence, sentence\n
        ...
    """
    files = [filepath+x for x in os.listdir(filepath)]
    source = []
    target = []
    #get the supervised training data into a two lists 
    for i in range(len(files)):
        df = pd.read_csv(files[i], header=None)
        source.extend(list(df[0]))
        target.extend(list(df[1]))
    
    #replacing the special tokens in supervised targets and source 
    if specialtokens!="":
        replacement_tokens = open(specialtokens, "r").read().split("\n")
        for i in range(len(target)):
            for j in range(len(replacement_tokens)):
                target[i] = target[i].replace(replacement_tokens[j], '')
                source[i] = source[i].replace(replacement_tokens[j], '')
    return source, target
    

def tokenize_unsupervised(source, size_tuple: tuple, extra_tokens: str, debug: int, tokenizer):
    #check if there is a previous tokenizer to use
    previous_tokenizer = True
    
    #no tokenizer then make a new one 
    if not tokenizer:
        previous_tokenizer = False
        tokenizer = T5TokenizerFast.from_pretrained(size_tuple[0])
        
        #add new tokens to tokenizer
        if (extra_tokens):
            tokenizer.add_tokens(open(extra_tokens, "r").read().split("\n"))
    
    
    model_inputs = []
    vocab = len(tokenizer.get_vocab())
    
    #for debugging you can just add tokenize 100 examples to quickly iterate
    tokenization_length = 0
    if bool(debug):
        tokenization_length = 100
    else:
        tokenization_length = len(source)
    
    
    for i in tqdm(range(tokenization_length)):
        #tokenize it 
        test = tokenizer(source[i], return_tensors='pt').input_ids[0]
        
        #split the tokenized training example into chunks the size of max input for the model
        chunks = test.split(size_tuple[2])
        for j in range(len(chunks)):
            tokens = chunks[j]
            #sentinel mask the training inputs 
            random = torch.rand(tokens.size(0))
            noise_mask = torch.where(random<=0.15, True, False)
            input_id=noise_span_to_unique_sentinel(tokens, noise_mask, vocab)
            label = nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab)
            
            #pad inputs and labels
            input_id = torch.cat([input_id, torch.zeros(size_tuple[2]-input_id.size(0), dtype=input_id.dtype)], dim=0)
            label = torch.cat([label, torch.zeros(size_tuple[2]-label.size(0), dtype=label.dtype)], dim=0)
            model_inputs.append({'input_ids':input_id.tolist(), 'labels':label.tolist()})
    print('Ending Tokenization of training data')
    
    print('Starting Dataset Creation')
    #create dataset 
    data_frame = pd.DataFrame.from_records(model_inputs)
    dataset = Dataset.from_pandas(data_frame).select(range(len(model_inputs)))
    dataset.set_format(
        type="torch", columns=["input_ids", "labels"],
    )
    if not previous_tokenizer:
        return [dataset], tokenizer
    else:
        return [dataset]
    

def tokenize_supervised(source: list[str], target: list[str], size_tuple: tuple, extra_tokens: str, debug: int, tokenizer):
    previous_tokenizer = True
    
    #check for a previous tokenizer make the new tokenizer
    if not tokenizer:
        previous_tokenizer = False
        tokenizer = T5TokenizerFast.from_pretrained(size_tuple[0])
        
        #add the extra tokens to tokenizer
        if (extra_tokens):
            tokenizer.add_tokens(open(extra_tokens, "r").read().split("\n"))

    model_inputs = []
    vocab = len(tokenizer.get_vocab())

    tokenization_length = 0
    if bool(debug):
        tokenization_length = 100
    else:
        tokenization_length = len(target)
    model_inputs = []
    for i in tqdm(range(tokenization_length)):
        
        #tokenize the inputs and target to max_length of size of model
        tokenized_input = tokenizer(source[i], padding='max_length', truncation=True, max_length=size_tuple[2])
        tokenized_target = tokenizer(target[i], padding='max_length', truncation=True, max_length=size_tuple[2])
        model_inputs.append({
            'input_ids': tokenized_input['input_ids'],
            'attention_mask': tokenized_input['attention_mask'],
            'labels': tokenized_target['input_ids'],
            'target_mask': tokenized_target['attention_mask'],
        })

    #create the dataset 
    data_frame = pd.DataFrame.from_records(model_inputs)
    dataset = Dataset.from_pandas(data_frame).select(range(len(model_inputs)))
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"],
    )
    if not previous_tokenizer:
        return [dataset], tokenizer
    else:
        return [dataset]

def main(args):
    #size dictionary with the number of LM heads and size of input
    size_dict = {
        0: ('t5-small', 6, 512),
        1: ('t5-base', 12, 768),
        2: ('t5-large', 24, 1024),
        3: ('t5-3b', 24, 1024),
        4: ('t5-11b', 24, 1024)
    }
    
    model = None
    training_args = None
    dataset = None
    tokenizer = None
    
    #load the args 
    with open(args.json_path) as f:
        data = json.load(f)
    
    #parse the args 
    parse_json(data)
    
    #get the size tuple from the data
    size_tuple = size_dict[data['size']]
    debug = data['debug']
    
    #loop through the training the methods 
    for i in range(len(data['training'])):
        
        #running the unsupervised running
        if list(data['training'][i].keys())[0]=="unsupervised":
            
            #get specific data area
            data_segment = data['training'][i]["unsupervised"]
            
            #get replacement tokens
            replacement_tokens = data_segment["replacement_tokens"]
            
            #get training data
            filepath = data_segment["data_path"]
            
            #process unsupervised data
            unsupervised_target = process_unsupervised_files(filepath, replacement_tokens)
            
            #get additional tokens for tokenizer
            extra_tokens = data_segment['extra_tokens']
            
            #check if the model has been trained before
            #use previous iterations tokenizer
            if data_segment['train_consec'] == False and len(data['training'])> 1:
                print("Starting unsupervised training after another training")
                dataset = tokenize_unsupervised(source, size_tuple, extra_tokens, debug, tokenizer)
                dataset = dataset[0]
            else:
                #set up model, dataset, and tokenizer 
                if data_segment['previously_trained'] == True:
                    #load previous model
                    print("Starting unsupervised training on loaded model")
                    dataset, tokenizer = tokenize_unsupervised(unsupervised_target, size_tuple, extra_tokens, debug, None)
                    dataset = dataset[0]
                    model = T5ForConditionalGeneration.from_pretrained(data_segment['previously_trained_model'])
                else:
                    print("Starting unsupervised training from scratch")
                    dataset, tokenizer = tokenize_unsupervised(unsupervised_target, size_tuple, extra_tokens, debug, None)
                    dataset = dataset[0]
                    model = T5ForConditionalGeneration.from_pretrained(size_tuple[0])
                print("Resizing embeddings")
                
                #resizes embeddings for extra tokens
                model.resize_token_embeddings(len(tokenizer))
                model.parallelize()
            print('Setting up Model')
            
            #checking if output dir exists and creates it if it doesnt
            output_dir = data_segment['model_path']
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            
            #setting up Huggingface training args
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=data_segment['epochs'],
                per_device_train_batch_size=data_segment['batch_size'],
            # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
                prediction_loss_only=True, # If I need co compute only loss and not other metrics, setting this to true will use less RAM
                learning_rate=0.001,
                # Run evaluation every eval_steps
                #save_steps=1000, # How often to save a checkpoint
                save_total_limit=1, # Number of maximum checkpoints to save
                remove_unused_columns=True, # Removes useless columns from the dataset
                run_name='run_name', # Wandb run name
                #logging_steps=1000, # How often to log loss to wandb
                # How often to run evaluation on the val_set
                #logging_first_step=False, # Whether to log also the very first training step to wandb
                #load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
                #metric_for_best_model="loss", # Use loss to evaluate best model.
                #greater_is_better=False # Best model is the one with the lowest loss, not highest.
            )
        elif list(data['training'][i].keys())[0]=="supervised":
            
            #getting segement of json for supervised
            data_segment = data['training'][i]["supervised"]
            
            #get supervised training data file path 
            filepath = data_segment["data_path"]
            
            #get the file path of tokens to replace in your training data 
            replacement_tokens = data_segment["replacement_tokens"]
            
            #process the source and target for supervised training
            supervised_source, supervised_target = process_supervised_files(filepath, replacement_tokens)
            extra_tokens = data_segment['extra_tokens']
            
            #check if the model has been trained before
            #use previous iterations tokenizer
            if data_segment['train_consec'] == False and len(data['training'])> 1:
                dataset = tokenize_supervised(supervised_source, supervised_target, size_tuple, extra_tokens, debug, tokenizer)
                dataset = dataset[0]
            else:
                #set up model, dataset, and tokenizer 
                if data_segment['previously_trained'] == True:
                    #load previous model
                    print("Starting unsupervised training on loaded model")
                    dataset, tokenizer = tokenize_supervised(supervised_source, supervised_target, size_tuple, extra_tokens, debug, tokenizer)
                    dataset = dataset[0]
                    model = T5ForConditionalGeneration.from_pretrained(data_segment['previously_trained_model'])
                else:
                    dataset, tokenizer = tokenize_supervised(supervised_source, supervised_target, size_tuple, extra_tokens, debug, tokenizer)
                    dataset = dataset[0]
                    model = T5ForConditionalGeneration.from_pretrained(size_tuple[0])
                    
                #resizing embeddigns for extra tokens
                model.resize_token_embeddings(len(tokenizer))       
                #parallel 
                model.parallelize()
            print('Setting up Model')
            output_dir = data_segment['model_path']
            
            #check if the output dir doesnt exists and make it 
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            
            #set up the huggingface training args 
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=data_segment['epochs'],
                per_device_train_batch_size=data_segment['batch_size'],
            # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
                prediction_loss_only=True, # If I need co compute only loss and not other metrics, setting this to true will use less RAM
                learning_rate=0.001,
                # Run evaluation every eval_steps
                #save_steps=1000, # How often to save a checkpoint
                save_total_limit=1, # Number of maximum checkpoints to save
                remove_unused_columns=True, # Removes useless columns from the dataset
                run_name='run_name', # Wandb run name
                #logging_steps=1000, # How often to log loss to wandb
                # How often to run evaluation on the val_set
                #logging_first_step=False, # Whether to log also the very first training step to wandb
                #load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
                metric_for_best_model="loss", # Use loss to evaluate best model.
                #greater_is_better=False # Best model is the one with the lowest loss, not highest.
            )
        print(f"Prepping Trainer {i+1}")
        
        #preparing the trainer for running
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        print('Training Model')
        #train model
        trainer.train()
        
        #deparallelize the model in order to save it 
        model.deparallelize()
        print('Saving Model')
        
        #save the model
        trainer.save_model(output_dir + '/model')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train all sizes of T5 on all types of data')

    parser.add_argument('--json_path', '-j', required=True, type = str, help='path to json settings')

    args = parser.parse_args()
    main(args)
