#import necessary libraries
import csv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
import openai
from RAG import RAG

#initialize 3 testing quereies
querries = ["What role can AI play in predicting mental health crises before they occur, and how can we validate such predictions ethically?", "How can multimodal AI systems (text, voice, facial expression) be integrated to provide more accurate mental health assessments?", "How can AI contribute to large-scale mental health research without violating individual consent or autonomy?"]

#initialize database container
database = []

#initialize database data
with open("Database.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        database.append(row[2])
        
#initialize pre-trained embedding model
pre_trained_embedded = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code = True)

#initialize fine-tuned embedding model
fine_tuned_embedded = SentenceTransformer("Embedding_Model") #personal file path

#initialize api key
api = input("Please enter your API key: ")
openai.api_key = api

#initialize RAG for pre-trained
base_RAG = RAG(database, pre_trained_embedded, api, 5)

#initialize RAG for fine-tuned
fine_tuned_RAG = RAG(database, fine_tuned_embedded, api, 5)

#initialize output container
output = [["User Query", "Base RAG Output", "Fine-Tuned RAG Output", "Base Language Model Output"]]

#iterate through user queries
for question in querries:
    
    #run RAG pre-trained model
    response_1 = base_RAG.run(question)

    #run RAG fine-tuned model
    response_2 = fine_tuned_RAG.run(question)

    #run language model
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": question}],
        max_tokens = 300
    )
    response_3 = response["choices"][0]["message"]["content"].strip()

    #save each of the outputs from the frameworks to the container
    output.append([question, response_1, response_2, response_3])

#write the outptus to a csv file
with open("results.csv", "w", encoding = "utf-8", newline = '') as out_file:
    writer = csv.writer(out_file)
    writer.writerows(output)