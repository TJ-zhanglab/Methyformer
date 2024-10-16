from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
from torch import nn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Use Methylformer to make predictions.")
    parser.add_argument(
        "--tissue", type=str, default=None,
        help="Tissue predicted using Methlyformer.")
    parser.add_argument(
        "--predict_file", type=str, default=None,
        help="A csv file containing the predicted sequences.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-500m-1000g', add_prefix_space=True)  
    model_checkpoint = '/mnt/storage/raid/jiali/Pretrained-model/'+args.tissue+"/"
    #model-1000g-blood 
    #model-1000g-cns
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint) 

    def tokenize_dataset(dataset):
        return tokenizer(dataset["sequences"])    

    dataset = load_dataset("csv",data_files=args.predict_file)  
    dataset = dataset.map(tokenize_dataset, batched=True)    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = data_collator
        )
    
    prediction_log = trainer.predict(dataset['train'])
    predictions = np.argmax(np.array(prediction_log.predictions), axis=-1)
    
    pd.DataFrame(predictions,columns=["labels"]).to_csv(args.predict_file[:-4] + "_" + "prediction.csv",index=False)

if __name__ == "__main__":
    main()