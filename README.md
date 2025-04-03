# genai-health-consultant


# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/abubakarcool/genai-health-consultant.git
```
### STEP 01- Create a conda environment using Anaconda Prompt CLI

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### create requirments.txt file and execute below command in that same anaconda prompt cli
```bash
pip install -r requirements.txt
```
### run template.py it will create files and folders of the project auto


### we are developing a library or modular app structure so create setup.py and also add -e . to tell 
#### Install the current folder (.) as a Python package, but in editable mode, so changes to the code reflect immediately without reinstallation.
### again run below cmd
```bash
pip install -r requirements.txt
```
### now it will create medibot_AI_Project.egg-info folder as well and now pip list this package will also appear 

### Now lets push this to our github 
```bash
git add .
git commit -m "folder structure added"
git push origin main
```



### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone
