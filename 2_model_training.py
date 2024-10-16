import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_dataset,load_metric
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a classification task")
    parser.add_argument("--model_name", type=str, default=None,
                        help="The name of pre-trained model.")
    parser.add_argument("--train_file", type=str, default=None, 
                        help="A csv file containing the training data.")
    parser.add_argument("--test_file", type=str, default=None, 
                        help="A csv file containing the testing data.")
    parser.add_argument("--output_dir", type=str, default="./", 
                        help="Output dir.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size")
    parser.add_argument("--train_epochs", type=int, default=None, 
                        help="Number of training epochs")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_checkpoint =args.model_name
    #'InstaDeepAI/nucleotide-transformer-500m-human-ref'
    #'InstaDeepAI/nucleotide-transformer-500m-1000g'
    #'InstaDeepAI/nucleotide-transformer-2.5b-multi-species'
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                               cache_dir='/mnt/gpunode/diskArray/jiali/Nucl-Trans',
                                                               num_labels=2)   

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)  

    def tokenize_dataset(dataset):
        return tokenizer(dataset["sequences"])    

    dataset = load_dataset("csv",data_files={"train":args.train_file,
                                      "test":args.test_file})  

    dataset = dataset.map(tokenize_dataset, batched=True)     
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)    

    def compute_metrics(eval_preds):
        metric = load_metric("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        MCC = matthews_corrcoef(labels,predictions)
        #np.savetxt(args.log_name + str(MCC)+ ".csv", logits, delimiter="," )
        return {'MCC': MCC}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps = 20000,
        num_train_epochs=args.train_epochs,
        save_strategy="epoch", 
        push_to_hub=False
    )    

    if torch.cuda.is_available():
        model = model.cuda()

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()