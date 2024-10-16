from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer
from datasets import load_dataset,Dataset
import pandas as pd
import numpy as np
import torch
from torch import nn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Use Methylformer to make predictions.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model used for predicting.")
    parser.add_argument(
        "--cpgref", type=str,default=None,
	help="A csv file used for predicting containing 400bp sequences the cpg sites extracted from the reference genome.")
    parser.add_argument(
        "--snp", type=str,default=None,
        help="A csv file used for predicting containing the snps.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cpgref = pd.read_csv(args.cpgref)
    snp = pd.read_csv(args.snp)

    data_ref_mut = pd.DataFrame()
    x=0
    for i in range(0,len(cpgref)):
        ref_seq = cpgref.loc[i,"seq"]
        start = cpgref.loc[i,"position"].astype(int) - 199
        end = cpgref.loc[i,"position"].astype(int) + 200
        a = (cpgref.loc[i,'chr'] == snp["chr"]) & (snp['pos'] >= start) & (snp['pos'] <= end)
        snv_count = a.sum()
        if snv_count ==0:
            continue
        else:
            snv_inform = snp[a].reset_index(drop=True)
            for j in range(snv_count):
                change = snv_inform.loc[j,"pos"]
                seq_mut = ref_seq[:change-start] + snv_inform.loc[j,"alt"] +ref_seq[change-start+1:]
                data_ref_mut.loc[x,'cpg'] = cpgref.loc[i,'cpg']
                data_ref_mut.loc[x,'chr'] = cpgref.loc[i,'chr']
                data_ref_mut.loc[x,'cpg_pos'] = cpgref.loc[i,'position']
                data_ref_mut.loc[x,'snv_pos'] = snv_inform.loc[j,'pos']
                data_ref_mut.loc[x,'REF'] = snv_inform.loc[j,'ref']
                data_ref_mut.loc[x,'ALT'] = snv_inform.loc[j,'alt']
                data_ref_mut.loc[x,'seq_ref'] = cpgref.loc[i,'seq']
                data_ref_mut.loc[x,'seq_mut'] = seq_mut
                x+=1

    tokenizer = AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-500m-1000g', 
                                              add_prefix_space=True)  
    model_checkpoint = '/mnt/storage/raid/jiali/Pretrained-model/'
    model = AutoModelForSequenceClassification.from_pretrained('/mnt/storage/raid/jiali/Pretrained-model/'+args.model+"/")

    def tokenize_dataset(dataset):
        return tokenizer(dataset["seq"])

    dataset_ref = Dataset.from_pandas(data_ref_mut[['seq_ref']])
    dataset_mut = Dataset.from_pandas(data_ref_mut[['seq_mut']])
    dataset_ref = dataset_ref.map(lambda x: tokenize_dataset({"seq": x["seq_ref"]}), batched=True)
    dataset_mut = dataset_mut.map(lambda x: tokenize_dataset({"seq": x["seq_mut"]}), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
            model = model,
            tokenizer = tokenizer,
            data_collator = data_collator
            )
    
    predictions_ref = trainer.predict(dataset_ref)
    predictions_mut = trainer.predict(dataset_mut)

    logits_ref = predictions_ref.predictions
    logits_mut = predictions_mut.predictions

    predictions_ref = np.argmax(np.array(logits_ref), axis=-1)
    predictions_mut = np.argmax(np.array(logits_mut), axis=-1)

    data_ref_mut['predicted_ref'] = predictions_ref 
    data_ref_mut['predicted_mut'] = predictions_mut

    data_ref_mut["delta"] = data_ref_mut['predicted_mut']-data_ref_mut['predicted_ref']

    data_ref_mut.to_csv(args.model + "_result.csv",index = False)

if __name__ == "__main__":
    main()
