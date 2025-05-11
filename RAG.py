#import necessary libraries
import torch.nn.functional as functional
import openai
import math

#initialize RAG Framework class
class RAG:

    #RAG Framework constructor
    def __init__(self, database, embedding_model, api, k):

        #initialize k value
        self.k = k

        #initialize embedding model from passed model
        self.embedding_model = embedding_model

        self.database = []

        #embed the database
        for text in database:
            embedding = self.embedding_model.encode(text, convert_to_tensor = True)
        
            #save values as matrix (vector database)
            self.database.append([text, embedding])

        #initialize API key
        self.api = api

    #run function
    def run(self, user_query):

        #embed user query
        embed_user = self.embedding_model.encode(user_query, convert_to_tensor = True)

        #initialize container
        similarities = []
        
        #run comparison function against each data value in database
        for item in self.database:
            score = functional.cosine_similarity(embed_user, item[1], dim = 0)
            similarities.append([score, item[0]])

        #compute the top k matches
        similarities = sorted(similarities, key = lambda element: element[0], reverse = True)
        top_k = similarities[:self.k]

        #initialize container
        prompt = ''
        
        #use saved values to determine the text to help prompt
        for element in top_k:
            prompt = prompt + element[1] + "\n"
        messages = [{"role":"user", "content": prompt + user_query}]
        
        #generate response
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages,
            max_tokens = 300
        )
        response = response["choices"][0]["message"]["content"].strip()
        
        #return model output
        return response

