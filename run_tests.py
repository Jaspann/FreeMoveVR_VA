import sys, chromadb, ollama, json

test_questions = [
    "What criteria does the driver use to update the pose when a bluetooth landmark message is received?",
    "How should the mobile camera be positioned in the play space according to the user instructions?",
    "When does the success sound play on desktop?",
    "When does the failure sound play on mobile?",
    "What does the failure sound imply on desktop?",
    "Can I swap the camera between front and back on mobile?",
    "Can I swap cameras on desktop?",
    "What is the disconnect message flag in a Bluetooth Message?",
    "How long does calibration take on mobile?",
    "How many landmarks are in a pose on desktop?",
    "What is the max number of people that can be tracked using the pose detector on desktop?",
    "Does the desktop app support Mac?",
    "Does the desktop app support Windows?",
    "Does the desktop app support Linux?",
    "Does the desktop app support FreeBSD?",
    "What UI framework does the desktop app use?",
    "Is using named pipes a valid connection method in the driver?",
    "What connection method is available on mobile?",
    "What pose detection model is used on desktop?",
    "Can I add a device whitelist to the driver?",
    "How is rotational information represented in the driver?",
    "What is the list of trackers that the driver emulates?",
    "Is there UI to set my height in the mobile app?",
    "Is there UI to set my height in the desktop app?",
    "Is there UI to set my height in the driver?",
    "Can I set the active trackers on mobile?",
    "Can I set the active trackers on desktop?",
    "On desktop what suggestions does it make if you are not found during calibration?",
    "What happens when a new calibration message is received in the driver?",
    "What happens when a new calibration message is received on mobile?",
]

Q_PREFIX = "Represent this sentence for searching relevant passages: "

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

initial_data = []
with open('data.json', 'w') as file:
    json.dump(initial_data, file, indent=4)

for query in test_questions:
    database_prompt = f"QUESTION: {query}\n\n INSTRUCTIONS: Based on the QUESTION provided, determine the platform that the question is asking about. If the QUESTION is asking about a mobile app or Flutter, return `MOBILE`, if it asks about a desktop app or Python, return `DESKTOP`, and if it asks about the driver or C++, return `DRIVER`. If there is no platform specified, return `ANY`. Your response should only include the answer. Do not provide any further explanation."
    database_value = ollama.generate(model="llama3.1", prompt=database_prompt, stream=False)['response'].lower()
    # database_value = ollama.generate(model="granite3-dense:8b", prompt=database_prompt, stream=False)['response'].lower()

    prefixed_query = f"{Q_PREFIX}" + query

    print(prefixed_query)
    query_embed = ollama.embed(model="snowflake-arctic-embed", input=prefixed_query)['embeddings']

    if 'mobile' in database_value:
        collection = chroma_client.get_or_create_collection(name="mobile_class")
        # collection = chroma_client.get_or_create_collection(name="mobile_code")
        related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
    elif 'desktop' in database_value:
        collection = chroma_client.get_or_create_collection(name="desktop_class")
        # collection = chroma_client.get_or_create_collection(name="desktop_code")
        related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
    elif 'driver' in database_value:
        collection = chroma_client.get_or_create_collection(name="driver_class")
        # collection = chroma_client.get_or_create_collection(name="driver_code")
        related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
    else:
        collection = chroma_client.get_or_create_collection(name="mobile_class")
        # collection = chroma_client.get_or_create_collection(name="mobile_code")
        related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=2)['documents'][0])
        collection = chroma_client.get_or_create_collection(name="desktop_class")
        # collection = chroma_client.get_or_create_collection(name="desktop_code")
        related_docs += '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=2)['documents'][0])
        collection = chroma_client.get_or_create_collection(name="driver_class")
        # collection = chroma_client.get_or_create_collection(name="driver_code")
        related_docs += '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=2)['documents'][0])



    related_docs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=5)['documents'][0])
    prompt = f"CODE:\n{related_docs}\n\nQUESTION:\n{query}\n\nINSTRUCTIONS: Answer the users QUESTION using the CODE snippets above. The CODE snippets are based off the platform in the question. Keep your answer ground in the facts of the CODE. If the CODE doesn't directly relate to the QUESTION, try your best to answer with the information provided."

    output = ollama.generate(model="llama3.1", prompt=prompt, stream=False)
    # output = ollama.generate(model="granite3-dense:8b", prompt=prompt, stream=False)

    print("\n\n\n\n" + prompt + "\n\n\n")

    print(f"Answer: {output['response']}")

    json_object = {
        'question': query,
        'database_value': database_value,
        'related_docs': related_docs,
        'output': output['response']
    }

    try:
        with open('data.json', 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    
    data.append(json_object)
    
    with open('data.json', 'w') as file:
        json.dump(data, file, indent=4)
