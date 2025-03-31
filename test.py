from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(model="text-embedding-ada-002",
input="Your chunk of text here")
embedding_vector = response.data[0].embedding