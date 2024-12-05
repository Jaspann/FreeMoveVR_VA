import sys, chromadb, ollama

Q_PREFIX = "Represent this sentence for searching relevant passages: "

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

query = " ".join(sys.argv[1:])

database_prompt = f"QUESTION: {query}\n\n INSTRUCTIONS: Based on the QUESTION provided, determine the platform that the question is asking about. If the QUESTION is asking about a mobile app or Flutter, return `MOBILE`, if it asks about a desktop app or Python, return `DESKTOP`, and if it asks about the driver or C++, return `DRIVER`. If there is no platform specified, return `ANY`. Your response should only include the answer. Do not provide any further explanation."
database_value = ollama.generate(model="llama3.1", prompt=database_prompt, stream=False)['response'].lower()

prefixed_query = f"{Q_PREFIX}" + query

print(prefixed_query)
query_embed = ollama.embed(model="snowflake-arctic-embed", input=prefixed_query)['embeddings']

if 'mobile' in database_value:
    collection = chroma_client.get_or_create_collection(name="mobile_class")
    related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
elif 'desktop' in database_value:
    collection = chroma_client.get_or_create_collection(name="desktop_class")
    related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
elif 'driver' in database_value:
    collection = chroma_client.get_or_create_collection(name="driver_class")
    related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
else:
    collection = chroma_client.get_or_create_collection(name="mobile_class")
    related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=2)['documents'][0])
    collection = chroma_client.get_or_create_collection(name="desktop_class")
    related_docs += '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=2)['documents'][0])
    collection = chroma_client.get_or_create_collection(name="driver_class")
    related_docs += '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=2)['documents'][0])



related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
prompt = f"CODE:\n{related_docs}\n\nQUESTION:\n{query}\n\nINSTRUCTIONS: Answer the users QUESTION using the CODE snippets above. The CODE snippets are based off the platform in the question. Keep your answer ground in the facts of the CODE. If the CODE doesn't directly relate to the QUESTION, try your best to answer with the information provided."

rag_output = ollama.generate(model="llama3.1", prompt=prompt, stream=False)

print("\n\n\n\n" + prompt + "\n\n\n")

print(f"Answered with RAG: {rag_output['response']}")