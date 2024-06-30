import sys
sys.path.append("../libs/")

import argparse
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from train_utils import train_with_anchoring, train_without_anchoring,Dataset, FeatureQuantization

from model import Area2Vec
import torch

def train_area2vec(input_path, batch_size, learning_rate, num_epochs, save_epoch, alpha, beta, weight_type, cuda):
    anchor_df = pd.read_csv("../data/anchor_data/anchor_df.csv")
    stay_df = pd.read_csv(input_path)
    quantization = FeatureQuantization()
    dataset = Dataset(quantization)
    dataset.gen_dataset(stay_df)
    dataset.gen_anchor_dataset(anchor_df)
    
    # load anchor embedding
    anchor_num = 512
    initial_embedding_weight = torch.rand(dataset.num_meshs, 8)
    anchor_embedding = torch.load("../data/anchor_data/anchor_embeddings.pth")
    initial_embedding_weight[-anchor_num:] = anchor_embedding

    # define model
    device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else 'cpu')
    model = Area2Vec(
        num_areas=dataset.num_meshs,
        embed_size=8,
        num_output_tokens=dataset.num_tokens,
        device=device
    )
    model.initialize_weights(embedding_weight=initial_embedding_weight, freeze_anchor_num=anchor_num)
    model = model.to(device)
    
    train_with_anchoring(
        model, 
        dataset,  
        save_path = "../output/sample_model/", 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        num_epochs=num_epochs, 
        save_epoch=save_epoch, 
        weight_type=weight_type, 
        alpha=alpha, 
        beta=beta
        )
    
    # # If you don't need anchoring
    # train_without_anchoring(
    #     model, 
    #     dataset,  
    #     save_path = "../output/sample_model/", 
    #     batch_size=batch_size, 
    #     learning_rate=learning_rate, 
    #     num_epochs=num_epochs, 
    #     save_epoch=save_epoch
    #     )
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_path', type=str, help='Input csv file path', default="../data/sample_data/stay_df.csv")
    parser.add_argument('--batch_size', type=int, help='Batchsize in training', default=1024)
    parser.add_argument('--learning_rate', type=float, help='Leaning rate in training', default=0.01)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs in training', default=200)
    parser.add_argument('--save_epoch', type=int, help='How many epochs to save the model', default=10)
    parser.add_argument('--alpha', type=float, help='Initial anchor power', default=0.3)
    parser.add_argument('--beta', type=float, help='Final anchor power', default=1.0)
    parser.add_argument('--weight_type', type=str, help='Weight function for anchor power', default="exponential")   
    parser.add_argument('--cuda', type=int, help='Cuda number to use', default=0) 
    args = parser.parse_args()
    train_area2vec(args.input_path, args.batch_size, args.learning_rate, args.num_epochs, 
         args.save_epoch, args.alpha, args.beta, args.weight_type, args.cuda)
    pass