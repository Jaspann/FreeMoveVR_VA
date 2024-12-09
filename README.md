# FreeMoveVR VA

This project creates a Virtual Assistant (VA) for a personal project of mine, FreeMoveVR. 
FreeMoveVR is an interesting use case for VAs, as it's code is split across three repositories, 
each using their own languages and different formats. 


The project was created using Python 3.11.3 on Mac,
and uses Ollama to run LLMs locally and Chroma for vector storage.

Download Ollama via their site: https://ollama.com/

Chroma requires Docker. Using Docker, create and start up Chroma with:

```bash
docker run -d
    -p 8000:8000
    -v chroma-data:/chromadb/data
    chromadb/chroma
```

Then download the required models:

```bash
ollama pull llama3.1
ollama pull granite3-dense:8b
ollama pull snowflake-arctic-embed
```

Create a venv and pip install `requirements.txt`

Place the zip file of FreeMoveVR into this folder, 
then run all cells in `load_documents.ipynb` then all cells in `load_documents.ipynb`. 

From there, tests can be run via `run_tests.py` swapping out the commented lines based on which test to preform, 
or interacted with via the `free_move_vr_va.py` file.