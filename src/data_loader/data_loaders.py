import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List

class OzonDataset(Dataset):
    def __init__(self, df, tabular_cols, target_col, id_col, text_cols):
        self.df = df
        self.tabular_cols = tabular_cols
        self.target_col = target_col
        self.id_col = id_col
        self.text_cols = text_cols
        
        self.texts = self.df[self.text_cols].apply(
            lambda row: ' [SEP] '.join(row.values.astype(str)), axis=1
        ).tolist()
            
        self.tabular_features = torch.FloatTensor(self.df[self.tabular_cols].values)
        self.labels = torch.FloatTensor(self.df[self.target_col].values)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_text = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )

        tabular = self.tabular_features[idx]
        label = self.labels[idx].unsqueeze(0)
        
        return {
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0),
            "tabular": tabular,
            "x_numerical": tabular,             
            "x_categorical": torch.empty(0, dtype=torch.int64), # No categories in data
        }, label


def get_dataloaders(train_df, val_df, images_dir, tokenizer, tabular_cols, target_col, id_col, text_cols, batch_size, num_workers=None, sampler=None):
    train_dataset = OzonDataset(
        df=train_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        target_col=target_col,
        id_col=id_col,
        text_cols=text_cols,
    )   
    val_dataset = OzonDataset(
        df=val_df,
        images_dir=images_dir,
        tokenizer=tokenizer,
        tabular_cols=tabular_cols,
        target_col=target_col,
        id_col=id_col,
        text_cols=text_cols,
    )
    if sampler:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, pin_memory=True, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return train_loader, val_loader

