#import necessary libraries
import pandas as pd
import openai
import time

#import data from the csv file
raw = pd.read_csv("RawDataFinalAssignment.csv", encoding = "utf-8", encoding_errors = "replace")

#database data
database = pd.DataFrame(columns = ["Paper Title", "Text"])
#training data
training = pd.DataFrame(columns = ["User Questions", "Text", "Paper Title"])

#initialize api key
openai.api_key = input("Please enter your API key: ")

#initialize prompt
prompt = "You are a student with a facination of the implications of advancements in AI on mental health. Ask a question about the above text that you would want elaboration on but is still answered or mentioned in the text."

#iterate through data assigning to database and training
for index, row in raw.iterrows():
    if row["Text"] == "":
        pass

    #add even values to database
    if index % 2 == 0: 
        database.loc[-1] = [row["Title"], row["Text"]]
        database.index = database.index + 1
        database = database.sort_index()

    #add odd values to training
    if index % 2 == 1:

        #loop tracker
        this = True

        while this:
            
            try:

                #remove non-utf-8 characters
                text = str(row["Text"]).encode("utf-8", errors = "replace").decode("utf-8")
                
                #query openai for question about data
                response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = [{"role":"user", "content": text}, {"role":"user", "content": prompt}],
                    max_tokens = 200
                )
        
                #create new initial row
                new_row = [response["choices"][0]["message"]["content"].strip(), row["Text"], row["Title"]]
        
                #update training data
                training.loc[-1] = new_row
                training.index = training.index + 1
                training = training.sort_index()

                this = False

            except Exception as e:
                print(f"Error on row{index}: {e}")
                time.sleep(20)
                this = True
              
#save training and database data to respective csv files
database.to_csv("Database.csv")
training.to_csv("Training_Data.csv")