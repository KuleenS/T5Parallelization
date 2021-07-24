import torch
import os
import argparse
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5TokenizerFast

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

def main(args):
    print('Starting Process')

    print('Starting Data loading and cleaning')
    data = './data/'

    data_files1 = ['tgt-train.txt',
                'tgt-val.txt',
                'tgt-test.txt']

    target = []
    for i in range(len(data_files1)):
        filename = data+data_files1[i]
        if filename.find('tgt')>=0:
            with open(filename) as f:
                target.extend(f.readlines())

    replacement = ['[',']','(',')','<PAR>']
    for i in range(len(target)):
        for j in range(len(replacement)):
            target[i] = target[i].replace(replacement[j], '')
    print('Ending Data loading and cleaning')

    print('Starting Tokenization of training data')
    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    tokenizer.add_tokens(['1CUI', '2CUI'])
    model_inputs = []
    vocab = len(tokenizer.get_vocab())

    tokenization_length = 0
    if bool(args.debug):
        tokenization_length = 100
    else:
        tokenization_length = len(target)

    for i in tqdm(range(tokenization_length)):
        test = tokenizer(target[i], return_tensors='pt').input_ids[0]
        chunks = test.split(768)
        for j in range(len(chunks)):
            tokens = chunks[j]
            random = torch.rand(tokens.size(0))
            noise_mask = torch.where(random<=0.15, True, False)
            input_id=noise_span_to_unique_sentinel(tokens, noise_mask, vocab)
            label = nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab)

            input_id = torch.cat([input_id, torch.zeros(768-input_id.size(0), dtype=input_id.dtype)], dim=0)
            label = torch.cat([label, torch.zeros(768-label.size(0), dtype=label.dtype)], dim=0)
            model_inputs.append({'input_ids':input_id.tolist(), 'labels':label.tolist()})
    print('Ending Tokenization of training data')
    
    print('Starting Dataset Creation')
    data_frame = pd.DataFrame.from_records(model_inputs)
    dataset = Dataset.from_pandas(data_frame).select(range(len(model_inputs)))
    dataset.set_format(
        type="torch", columns=["input_ids", "labels"],
    )
    print('Ending Dataset Creation')
    
    print('Initializing Model')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    model.resize_token_embeddings(len(tokenizer))

    print('Parallelizing Model')
    device_map = {}
    t5_heads_array = [0,1,2,3,4,5,6,7,8,9,10,11]
    for i in range(args.gpus):
        device_map[i] = t5_heads_array[i*(len(t5_heads_array)//args.gpus):(i+1)*(len(t5_heads_array)//args.gpus)]

    model.parallelize(device_map=device_map)

    print('Setting up Model')
    output_dir = './modeldump/T5BaseUnsup'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
       
      # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
        prediction_loss_only=True, # If I need co compute only loss and not other metrics, setting this to true will use less RAM
        learning_rate=0.001,
         # Run evaluation every eval_steps
        save_steps=1000, # How often to save a checkpoint
        save_total_limit=1, # Number of maximum checkpoints to save
        remove_unused_columns=True, # Removes useless columns from the dataset
        run_name='run_name', # Wandb run name
        logging_steps=1000, # How often to log loss to wandb
        # How often to run evaluation on the val_set
        logging_first_step=False, # Whether to log also the very first training step to wandb
        load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
        metric_for_best_model="loss", # Use loss to evaluate best model.
        greater_is_better=False # Best model is the one with the lowest loss, not highest.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    print('Training Model')
    trainer.train()

    print('Saving Model')
    trainer.save_model(output_dir + '/model')

    print('Finished Process')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Train T5 base unsupervised on MIMIC data ')

    parser.add_argument('--batch_size', '-b', required=True, type = int, help='batch size of model')

    parser.add_argument('--epochs', '-e', required=True, type = int, help='epochs trained for model')

    parser.add_argument('--gpus', '-g', required=True, type = int, help='gpus of system (THIS DOES NOT WORK FOR MULTIPLE CLUSTERS)')

    parser.add_argument('--debug', '-d', required=True, type = int, help='runs a quick tokenization process to test if everything is working')

    args = parser.parse_args()
    main(args)