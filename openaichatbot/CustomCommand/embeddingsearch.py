import openai
import pandas as pd
import os
import wget
from ast import literal_eval

# Chroma's client library for Python
import chromadb

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Loading DATA
# embeddings_url = 'enter-the-website-url'

# The file is ~700 MB so this will take some time
# wget.download(embeddings_url)

#Reading datas using zipfile incase file is a .zip file
import zipfile
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
    zip_ref.extractall("../data")

# storing data to a variable to 
article_df = pd.read_csv('../data/extracted-zip-file_name')

# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

#SETTING UP CHROMA

# chroma_client = chromadb.EphemeralClient() # Equivalent to chromadb.Client(), ephemeral.
# Uncomment for persistent client
chroma_client = chromadb.PersistentClient()

# Getting and checking openai api key
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

wikipedia_content_collection = chroma_client.create_collection(name='wikipedia_content', embedding_function=embedding_function)
wikipedia_title_collection = chroma_client.create_collection(name='wikipedia_titles', embedding_function=embedding_function)

# Populating collections

# Add the content vectors
wikipedia_content_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.content_vector.tolist(),
)

# Add the title vectors
wikipedia_title_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.title_vector.tolist(),
)


# SEARCHING FEATURES TO BE ADDED TO CUSTOM COMMAND

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],
                'title': dataframe[dataframe.vector_id.isin(results['ids'][0])]['title'],
                'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['text'],
                })
    
    return df

title_query_result = query_collection(
    collection=wikipedia_title_collection,
    query="modern art in Europe",
    max_results=10,
    dataframe=article_df
)
title_query_result.head()

content_query_result = query_collection(
    collection=wikipedia_content_collection,
    query="Famous battles in Scottish history",
    max_results=10,
    dataframe=article_df
)
content_query_result.head()

#ykik this is from where

from chromadb import chromadb_client
import openai

chromadb_client.setup(api_key='YOUR_CHROMADB_API_KEY')
openai.api_key = 'YOUR_OPENAI_API_KEY'

def search_with_embeddings(search_string):
    import openai
    import pandas as pd
    import os
    import wget
    from ast import literal_eval

# Chroma's client library for Python
    import chromadb
    EMBEDDING_MODEL = "text-embedding-ada-002"
    article_df = pd.read_csv('../data/extracted-zip-file_name')
    # Query ChromaDB for embeddings
    embeddings = chromadb_client.query(search_string)

    # Use OpenAI to perform similarity search
    openai_result = openai.Completion.create(
        engine="text-davinci-002",
        prompt=search_string,
        n=1,
        temperature=0,
    )

    # Extract relevant information from OpenAI result
    openai_embedding = openai_result['choices'][0]['logit_blobs']

    # Your logic to compare embeddings and decide if the search string is a match
    # Example: Check if the similarity score is above a certain threshold
    similarity_threshold = 0.8
    if calculate_similarity(embeddings, openai_embedding) > similarity_threshold:
        return True
    else:
        return False

def calculate_similarity(embeddings1, embeddings2):
    # Your logic to calculate similarity between embeddings
    # Example: Cosine similarity
    # Adjust this based on the structure of your embeddings
    return 0.0  # Replace with your actual similarity calculation
