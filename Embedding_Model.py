#import necessary libraries
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader
import csv

#initialize training container
training = []

#initialize training data for fine-tuning
with open("Training_Data.csv", "r", encoding = "utf-8") as file:
    reader = csv.reader(file)

    #iterate through rows in csv file
    for row in reader:
        question = row[0]
        text = row[1]

        #save row with propper label and inputs to training container
        training.append(InputExample(texts=[question, text], label = 1.0))

#initialize training data in dataloader class format
train_dataloader = DataLoader(training, shuffle = True, batch_size = 10)

#initialized pretrained model
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en')

#initialize loss function to be used in training
train_loss = losses.CosineSimilarityLoss(model)

#fine tune the model
model.fit(train_objectives = [(train_dataloader, train_loss)],
          epochs = 5,
          output_path = "Embedding_Model")