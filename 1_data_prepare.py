import os
import argparse
import pandas as pd
import numpy as np
import pysam
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Processing datasets for model training")
    parser.add_argument(
        "--betavalue",  type=str,default=None,
        help="A csv file containing methylation beta values.")
    parser.add_argument(
        "--cpgref", type=str,default=None,
	help="A csv file containing 400bp sequences the cpg sites extracted from the reference genome.")
    parser.add_argument(
        "--snpvcf", type=str,default=None,
	help="A vcf file containing the snps of the samples.")
    parser.add_argument(
        "--out_dir", type=str, default=None, help="A file path for outputfiles.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    #读取甲基化beta值，转换为二分类
    betavalue = pd.read_csv(args.betavalue, header=0)  
    betavalue.iloc[:,1:] = np.where(betavalue.iloc[:,1:] >= 0.5, 1, 0)
    sample_list = betavalue.columns[1:]

    #cpg附近的参考序列
    cpgref = pd.read_csv(args.cpgref, header=0) 

    #snp的信息
    vcf = pysam.VariantFile(args.snpvcf)

    snpdata = []
    for variant in vcf:
        variant_info = {
            "Chr": variant.chrom,
            "Pos": variant.pos,
            "Ref": variant.ref,
            "Alt": variant.alts[0] if variant.alts else None,
        }
        
        # 提取每个样本的基因型信息
        for sample in variant.samples:
            sample_info = variant.samples[sample]
            genotype = '/'.join(map(str, sample_info['GT']))  # 将元组转换为字符串
            variant_info[sample] = genotype  # 将基因型添加到变异信息中

        snpdata.append(variant_info)

    # 将列表转换为 DataFrame
    snpdata = pd.DataFrame(snpdata)

    os.makedirs("./temp_files/", exist_ok=True) 
    #替换snv
    for sample in sample_list:
        df = snpdata[['Chr', 'Pos', 'Ref', 'Alt', sample]]
        df=df[df[sample]!="0/0"]
        data = cpgref
        data["seq_mut"] = pd.Series(dtype='object')  
        for i in range(len(data)):
            ref_seq = data.loc[i,"seq"]
            start = data.loc[i,"position"].astype(int) - 199
            end = data.loc[i,"position"].astype(int) + 200
            a = (data.loc[i,'chr'] == df["chr"]) & (df['Pos'] >= start) & (df['Pos'] <= end)
            snv_count = a.sum()
            if snv_count ==0:
                continue
            else:
                snv_inform = df[a].reset_index(drop=True)
                seq_tmp = data.loc[i,"seq"]
                for j in range(snv_count):
                    change = snv_inform.loc[j,"Pos"]
                    seq_tmp = seq_tmp[:change-start] + snv_inform.loc[j,"Alt"][0] + seq_tmp[change-start+1:]
                seq_mut = seq_tmp
                data.loc[i,"seq_mut"] = seq_mut
                data.loc[i,"mut_cot"] = snv_count
        data.to_csv('./temp_files/'+str(sample)+'_snv.csv', index=False)


    #合并得到dataset
    data_0 = pd.read_csv('./temp_files/'+str(sample_list[0])+'_snv.csv' ,header = 0)
    data_0['seq_mut'] = data_0['seq_mut'].fillna(data_0['seq'])
    data_0=data_0.merge(betavalue[["cpg",sample_list[0]]],on="cpg")
    data_0=data_0[["seq_mut",sample_list[0]]]

    for i in range(1,len(sample_list)):
        data_2 = pd.read_csv('./temp_files/'+str(sample_list[i])+'_snv.csv' ,header = 0)
        data_2['seq_mut'] = data_2['seq_mut'].fillna(data_2['seq'])
        data_2 = data_2.merge(betavalue[["cpg",sample_list[i]]],on="cpg")
        data_2 = data_2[["seq_mut",sample_list[i]]]
        data = data_0.merge(data_2, how ='outer', on='seq_mut')
        data = data.reset_index(drop=True)


    data['labels'] = data.iloc[:,1:].apply(
        lambda row: row.dropna().unique()[0] if len(row.dropna().unique()) == 1 else 'delete',
        axis=1
    )

    dataset = data[["seq_mut","labels"]]
    dataset = dataset[dataset["labels"]!='delete']
    dataset.columns=["sequences","labels"]
    dataset.to_csv(args.out_dir+"dataset.csv",index=False)

    #分成训练集和测试集
    X = dataset['sequences']
    y = dataset['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    train["labels"] = train["labels"].astype(int)
    test["labels"] = test["labels"].astype(int)

    train.to_csv(args.out_dir+'train.csv',index= False)
    test.to_csv(args.out_dir+'test.csv',index= False)

if __name__ == "__main__":
    main()
