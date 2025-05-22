# CORD-19 QA Bot

This project implements a Question Answering (QA) bot specialized for the CORD-19 dataset, a collection of research papers on COVID-19 and related coronaviruses. The bot aims to help users quickly find answers to their questions from this vast corpus of scientific literature.

It utilizes a document retrieval component based on BM25 and a document reader component based on a pre-trained BERT model (specifically, `bert-large-uncased-whole-word-masking-finetuned-squad`) to find and extract relevant answers.

## Project Setup and Dependencies

### Prerequisites
- Python 3.x
- pip (Python package installer)
- Git

### Installation

1.  **Clone the repository:**
    Replace `<repository-url>` with the actual URL of this repository.
    ```bash
    git clone <repository-url>
    cd CORD-19-QA-Bot 
    ```
    (If you cloned it under a different name, `cd` into that directory).

2.  **Install Python dependencies:**
    The primary dependencies are listed in `requirements.txt`. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    The project also requires the `rank_bm25` library, which can be installed directly from GitHub:
    ```bash
    pip install git+https://github.com/dorianbrown/rank_bm25.git
    ```

3.  **Download NLTK resources:**
    Some scripts and notebooks use `nltk` for tasks like sentence tokenization. You'll need to download the 'punkt' tokenizer models. This can be done by running the following Python snippet:
    ```python
    import nltk
    nltk.download('punkt')
    ```

4.  **BERT Model:**
    - The system uses a pre-trained BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`) from the Hugging Face `transformers` library.
    - This model will be downloaded automatically (and cached) by the `transformers` library when the `DocReader` class is initialized for the first time (i.e., when you run the QA bot or relevant notebooks).

### Full List of Identified Python Libraries:
(This list includes dependencies from `requirements.txt` and other imported libraries)
- Flask==1.1.1
- flask_cors==3.0.8
- joblib==0.13.2
- pandas==0.25.0
- prettytable==0.7.2
- transformers>=2.3.0
- scikit_learn==0.21.2
- tika==1.19
- torch>=1.4.0
- markdown==3.1.1
- tqdm>=4.36.1
- wget==3.2
- pytorch_pretrained_bert>=0.6.2
- nltk
- rank_bm25
- tensorflow (Note: Tensorflow is not directly imported in most user-facing scripts but appears as a dependency or is used by underlying libraries like Hugging Face transformers. Ensure it is installed if errors related to it occur.)

## Data Processing

The CORD-19 dataset needs to be processed before it can be used by the QA bot. This involves parsing the JSON files containing the research papers and consolidating them into a structured format.

1.  **Download the CORD-19 Dataset:**
    You will need to download the CORD-19 dataset (e.g., from Kaggle). The dataset typically includes subdirectories like `biorxiv_medrxiv`, `comm_use_subset`, `noncomm_use_subset`, and `custom_license`. Place these folders into a parent directory, for example, `input/CORD-19-research-challenge/`.

2.  **Run the Processing Script:**
    The script `submission/data_process.py` (or the notebook `CORD19_Data_process.ipynb`) handles the data preprocessing.
    - It iterates through the JSON files in the dataset folders.
    - It extracts relevant information such as title, authors, and the body text.
    - The body text from each paper is concatenated.
    - This processed data is then saved into a CSV file named `covid_corpus.csv` in the project's root directory. This CSV is used by the document retriever and reader.

    To run the script, provide the path to the parent directory containing the CORD-19 dataset subfolders:
    ```bash
    python submission/data_process.py -i path/to/your/CORD-19-research-challenge/
    ```
    Example:
    ```bash
    python submission/data_process.py -i input/CORD-19-research-challenge/
    ```
    If `covid_corpus.csv` is not found when running the main QA bot, the bot will attempt to generate it automatically using this script, prompting for the dataset location.

    The `CORD19_Data_process.ipynb` notebook provides a step-by-step guide through this process and can also be used to generate the `covid_corpus.csv` file.

## QA Pipeline

The question answering process involves two main stages:

1.  **Document Retrieval:**
    - The `helper.DocRetriever` class (implemented in `helper/DocRetriever.py` and used by `submission/doc_retriever.py`) is responsible for this stage.
    - It uses the BM25Okapi algorithm, a ranking function based on Term Frequency-Inverse Document Frequency (TF-IDF), to find the most relevant documents (research paper paragraphs) from `covid_corpus.csv` for a given user query.
    - It tokenizes the corpus and the query, then calculates BM25 scores to identify the top N most relevant documents.

2.  **Answer Extraction (Document Reading):**
    - The `helper.DocReader` class (implemented in `helper/DocReader.py` and used by `submission/doc_reader.py`) handles this part.
    - It takes the top N documents retrieved by `DocRetriever` and the user's query as input.
    - It employs the pre-trained BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`).
    - For each document, it tokenizes the query and the document's sentences.
    - The BERT model then predicts the start and end tokens of the most likely answer within each sentence.
    - The system aggregates these potential answers and selects the best one based on the model's confidence scores.

The main script `submission/cord19_QA_bot.py` (and the notebook `CORD19_QA_bot.ipynb`) orchestrates this pipeline: it takes a user query, retrieves relevant documents, extracts the best answer, and presents it along with context like the paper's title and authors.

## Usage Instructions

There are two primary ways to use the CORD-19 QA Bot:

1.  **Running the Python Script:**
    The main script is `submission/cord19_QA_bot.py`.
    - Ensure you have processed the data and `covid_corpus.csv` exists in the root directory. If not, run the data processing script first (see "Data Processing" section).
    - To run the bot:
      ```bash
      python submission/cord19_QA_bot.py
      ```
    - If `covid_corpus.csv` is not present, the script will prompt you to enter the path to your CORD-19 dataset, for example:
      ```
      Dataset Folder is : path/to/your/CORD-19-research-challenge/
      Generating Corpus CSV File...
      ```
    - Once running, the script will prompt you to enter your query. Type your question and press Enter.
    - The bot will process the query and display the answer, title, authors, and the source paragraph.
    - Type `exit` to quit.

    Example interaction:
    ```
    Enter the query (type exit to quit) : what are the symptoms of covid-19
    Processing...
    query: what are the symptoms of covid-19

    answer: the most common symptoms include fever , cough , and myalgia or fatigue ; less common symptoms are sputum production , headache , haemoptysis , and diarrhoea

    title: Clinical features of patients infected with 2019 novel coronavirus in Wuhan, China

    authors: Huang Chaolin (Department of Pulmonary and Critical Care Medicine, ...), Wang Yeming (...), ... [List of authors truncated for brevity]

    paragraph: the most common symptoms include fever , cough , and myalgia or fatigue ; less common symptoms are sputum production , headache , haemoptysis , and diarrhoea .

    Enter the next query (type exit to quit) : exit
    Done!
    ```

2.  **Using the Jupyter Notebooks:**
    - The `CORD19_QA_bot.ipynb` notebook provides an interactive way to step through the QA process. Run the cells to see how queries are processed. This is useful for experimentation.

## Jupyter Notebooks

The repository includes several Jupyter notebooks:

-   **`CORD19_Data_process.ipynb`**:
    -   Details the processing of raw CORD-19 JSON data into `covid_corpus.csv`. Good for understanding data extraction/cleaning. Can be used to generate the corpus interactively.

-   **`CORD19_QA_bot.ipynb`**:
    -   Demonstrates the end-to-end QA pipeline: loading the corpus, initializing `DocRetriever` and `DocReader`, and answering sample queries. Useful for testing and understanding the system flow.

-   **`cord19_QA_opt.ipynb`**:
    -   Appears to explore optimizations or alternative approaches, possibly different tokenizers for BM25 or methods for document indexing/retrieval. May contain more advanced or experimental code.

These notebooks are valuable for deeper understanding and experimentation.

## Potential Improvements and Future Work

-   **Fine-tune BERT Model:** Further fine-tune the BERT model on the CORD-19 corpus for improved domain specificity.
-   **Advanced Retrieval Methods:** Explore dense vector embeddings (e.g., DPR, Sentence-BERT) for document retrieval.
-   **Corpus Pre-training for BERT:** Pre-train BERT on the CORD-19 text corpus (generatable via `data_process.py`) before QA fine-tuning.
-   **User Interface:** Develop a web interface (e.g., Flask, Streamlit) for easier access.
-   **Answer Validation/Confidence:** Implement confidence scores for answers.
-   **Scalability:** Optimize for larger datasets (ideas might be in `cord19_QA_opt.ipynb`).
-   **Multi-document Summarization:** Synthesize answers from multiple passages.
-   **Dependency Updates:** Regularly update models and libraries.
```
