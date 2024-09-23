# 🎙️ Political Debate Simulator
## 🏆 Winner of the 2024 Code the Vote Hackathon

The **Debate Bot** is an interactive web application designed to simulate a debate between two political candidates. Users can ask questions, read the candidates' responses, and vote on the best answers.

---

## ✨ Features

- **Interactive Debate**: Ask questions and receive responses from Candidate A and Candidate B.
- **Voting System**: Vote for the best response and see the overall results.
~~- -**Trivia Section**: Test your knowledge with trivia questions about US Presidents.~~
~~- **Earn Badges**: Earn badges for participation and correct answers in the trivia section.~~

---

## How to Use

1. **Visit WebApp**: Read a bit about each candidate and choose your moderator.
2. **Ask a Question or select a popular question**: Enter your question in the input box and click "Ask".
3. **Read Responses**: Review the responses from Candidate A and Candidate B.
4. **Vote**: Vote for the response you like best by clicking the "Vote" button under the candidate's response.
5. **See Winner**: See who you chose for each question asked.
6. **End Debate**: Click "End Debate" to see the overall results and the source for each answer.

~~## Getting Started~~

~~To run this project locally:~~

~~1. Clone the repository:~~

~~2. Navigate to directory:~~
   ~~cd presidential-debate-simulator~~
~~3. Create an API via openai.com and paste the apiKey into line 39 and 321 and click save.~~ 
   
~~4. Open index.html in your web browser. (Best using Chrome)~~

## Data Analysis and Visualizations
* View data visualizations created from mock_response_data.csv by running voting_viz.ipynb 
* Categorize raw questions into broader topics using categorize.py. This takes in questions from uncategorizedQuestions.csv and puts them through ChatGPT to classify them into categories and outputs these in a new csv. You will need to create an api key and adjust the code for this to work for you.


## 🔄 Updates

Stay informed about the latest changes and improvements to the project. Below you'll find a summary of recent updates:

> ### 08/07/2024
> 
> **Added Files:**
> 
> - `main.py`: This file includes functions from `rag_debate.py`. Use `rag_debate.py` to run inside a notebook. `main.py` requires a server.
> 
> **Purpose:**
> 
> Contains the main FastAPI application.
> 
> **Key Features:**
> 
> - Defines the FastAPI app instance.
> - Implements utility functions for text processing.
> - Includes the `QueryRequest` Pydantic model for request validation.
> - Sets up a startup event to preprocess and vectorize data.
> - Defines the `/query` endpoint to handle user queries and generate responses.
> - Uses OpenAI's GPT-4 model to generate responses based on retrieved text chunks.
---

> ### 08/11/2024
> 
> **Added Files:**
> 
> - `process_sources.py`: makes the chunks and vectorizes them and saves them into a pickle file. Also adjusted the chunking to enforce the chunks to stay within the set size
> 
> **Update:**
> 
> - `RAG Debate.py`
>   - remove the parts about processing the source files into vectors. This file is now only the user input and response generation
> 
> **Purpose:**
> 
> separates the vectorizing and chunking process to its own script so it could run apart from the game, which saves resources not reprocessing the files each game, and makes run time faster.
> Making the chunking enforce a size limit makes each chunk more focused so it helps the context retrieve relevant information 

---

> ### 08/13/2024
> 
> **Added Files:**
> 
> - `requirements.txt`
> 
> **Update:**
> 
> - `RAG Debate.py`
>   - Add - `main = 'insert model of choice'`
> 
> **Purpose:**
> 
> Ease of testing different models. As of today, `gpt-4o-mini` seems to be the best with responses.

---

> ### 08/21/2024
>
> **Renamed Files:**
>
> - `RAG Debate.py to TFID RAG Debate.py`
> - `process_sources.py to TFID_process_sources.py` Both changed to differentiate between TFID Vectorization and OpenAPI
>   
> **Added Files:**
> 
> - `OpenAPI RAG Debate.py`
> - `OpenAPI_process_sources.py`
> - `generated_client.zip`
> - `openai-api.yaml` - OpenAPI Spec File
> 
> **Purpose:**
> 
> To attempt using better vectorization via OpenAPI.
> 
> **Key Features:**
> 
> - API Client Configuration - Structured to interact with OpenAI's API, two specific models: `text-embedding-ada-002` for generating embeddings and `gpt-4o-mini` for generating chat completions. It sets up an API client using configuration classes and authorizes requests with an API key.
> - Text Preprocessing and Chunking - It includes functions for extracting, preprocessing, and chunking text data from .txt files, enabling efficient handling of large documents by breaking them into smaller, manageable pieces based on token length.
> - OpenAI Embedding Generation - The code provides a function to generate text embeddings using the `text-embedding-ada-002` model. These embeddings are crucial for comparing the similarity of different text chunks
> 
> **Requirements:**
>
> - JDK Development Kit 22.0.2 - https://www.oracle.com/java/technologies/downloads/?er=221886
> 1. In the Environment Variables window, under the System variables section, click New....
> 2. In the Variable name field, enter JAVA_HOME.
> 3. In the Variable value field, enter the path to your JDK installation directory. This will be something like: `C:\Program Files\Java\jdk-22.0.2`
> 4. Add the BIN to environment variables (path)
>
> - ONLY TO GENERATE NEW CLIENT - OpenAPI-generator -  https://repo1.maven.org/maven2/org/openapitools/openapi-generator-cli/7.7.0/openapi-generator-cli-7.7.0.jar - ONLY TO GENERATE NEW CLIENT - 
> 1. Ensure OpenAPI Spec File `openai-api.yaml` is located somewhere
> 2. Run the following - `java -jar C:\openapi-generator-cli.jar generate -i 'PATH TO "openai-api.yaml"' -g python -o ./generated-client`
> 3. You can install the generated client into your Python environment by navigating to the generated-client directory and running: `pip install .`
> 4. By default, the generated-client folder is created in the current working directory from which you run the OpenAPI Generator command.
> 

---

> ### 08/25/2024
> 
> **Update:**
> 
> - `TFID RAG Debate.py`
>   - Add RAGAS faithfulness and answer_relevancy scores for each query response
>   - Save the responses and scores to mysql
> 
> **Purpose:**
> 
>  The RAGAS scores are to keep a pulse on how good our chat bot answers are. These can be analyzed to help us improve the application.
>   

---

> ### 08/27/2024
> 
> **Added Files:**
> 
> - `suggested_questions.txt`
>   - Add sample questions into text file
> 
> **Purpose:**
> 
>  The sample questions will be the ones suggested to the users in the game. They will also be the baseline questions for which we will create a test set and validate that we have the right answers
> 

---

> ### 08/28/2024
> 
> **Added Files:**
> 
> - `debate_bot.py`
>   - The new main file of the chatbot. This combines what was `openapi_RAG Debate.py` with the updates to pull queries from and save responses to the mysql database.
>   - This also has a fix for retrieving bad texts by switching the "ascending" parameter of the sort to True
>
> - `sources/reichert/Reichert_WA_Seattletimes_Jul07_2024.txt` 
>   - start using document sources of our new candidates 
>   - old sources were moved into archive
> 
> **Update:**
> 
> - `OpenAPI__process_sources.py`
>   - folder path changed to use sources instead of archive

> ## 09/01/2024
>
> ### OpenAPI__process_sources.py
>
> #### Enhanced Preprocessing and Vectorization Function:
>
> - Updated the `preprocess_and_vectorize_combined` function to accept an additional `output_filename` parameter, allowing the flexibility to save vectorized chunks to specific files.
> - Combined the logic for preprocessing, chunking, and vectorizing text from different sources (e.g., Reichert and Ferguson) within the same function call.
>
> #### Modular and Flexible Main Execution Flow:
>
> - Refactored the `main()` function to handle multiple candidate datasets in a loop. This allows for processing multiple folders (e.g., "reichert" and "ferguson") in a single run.
> - Added a dictionary, `folder_paths`, to store folder paths for each candidate. This supports easy scaling and addition of more candidates if needed.
> - Dynamically sets output filenames for vectorized chunks using candidate names (e.g., `vectorized_chunks_reichert.pkl` and `vectorized_chunks_ferguson.pkl`).
>
> #### Improved Code Readability and Maintainability:
>
> - Removed redundant code and improved function and variable naming for better clarity.
> - Added informative print statements to provide feedback during processing, such as indicating the number of files processed and when vectorized chunks are saved successfully.
> - Consolidated similar logic to avoid code duplication, making the script easier to maintain and extend.
>
> #### Bug Fixes and Robustness Improvements:
>
> - Fixed the output file handling to avoid overwriting the same file. Each candidate's vectorized chunks are saved in separate files with clearly defined names.
> - Ensured compatibility with future expansions by making the code more modular and less error-prone through clear parameterization and usage of environment variables.
>
> #### Removal of Redundant Code:
>
> - Removed the duplicated block of code related to OpenAI client initialization and environment variable loading.
> - Cleaned up the script to avoid any redundant comments and consolidated similar code into reusable functions.
>
> ### debate_bot.py
>
> #### Refactored `find_best_texts` Function to Support Multiple Files:
>
> - The `find_best_texts` function has been updated to process multiple vectorized files provided as a list (`filenames`) instead of just a single hard-coded file. This allows it to handle different datasets more flexibly.
> - It iterates through each provided file, loading vectorized chunks and calculating cosine similarity between the query embedding and each chunk's embedding.
> - The function now returns the top `n` results based on similarity for each dataset.
>
> #### Separate Retrieval and Response Handling for Multiple Candidates:
>
> - Updated the `chatbot_with_prevectorized_chunks` function to retrieve texts separately for each candidate (Reichert and Ferguson). It calls `find_best_texts()` with different filenames (`['vectorized_chunks_reichert.pkl']` and `['vectorized_chunks_ferguson.pkl']`).
> - This separation ensures that responses are generated independently for each candidate using their specific contexts, preventing any mix-up between datasets.
>
> #### Enhanced Functionality for Saving Results:
>
> - Updated `save_to_csv` and `save_to_db` functions to handle and save results separately for each candidate. The updated code ensures that each candidate's response, context, and evaluation metrics are stored correctly in the database and CSV files.
>
> #### Refined the `generate_response` Function:
>
> - Modified the `generate_response` function to handle token length more precisely. If the generated prompt exceeds the maximum token limit, the context is truncated accordingly.
> - Corrected the prompt generation process to provide clearer and more relevant instructions to the language model.
>
> #### Improved Context Handling and Metrics Evaluation:
>
> - The code now generates responses and calculates metrics for each candidate independently, ensuring that the model's output is evaluated separately for Reichert and Ferguson.
> - Utilized `ragas` metrics (faithfulness and answer_relevancy) to score and evaluate the generated responses, which are then saved to the database and CSV files for each query.
>
---
> ## 09/05/2024
>
>
> #### 1. Added URL Extraction Functionality:
> - Introduced the `extract_url_from_txt()` function to extract URLs from the top of `.txt` files.
> - Validates if the first line is a valid URL and caches it for efficient access.
> - Handles scenarios where the `.txt` file is missing or contains an invalid URL.
>
> #### 2. Updated `find_best_texts` Function:
> - Modified to accept additional parameters: `pkl_filenames` and `txt_folder_path`.
> - Added logic to extract and cache URLs from corresponding `.txt` files to avoid redundant reads.
> - Returns a DataFrame that now includes URLs associated with each retrieved text chunk.
>
> #### 3. Enhanced Response Handling:
> - Added logic to manage responses separately for each candidate (Reichert and Ferguson).
> - Included source URLs in the print statements for better transparency of information sources.
> - Introduced a conditional check: if the response contains "I do not have enough information," the source URL is set to `"None"`.
>
> #### 4. Improved Code Modularity and Error Handling:
> - Removed hard-coded paths and added parameters to functions for better flexibility.
> - Improved error handling for file operations and missing data, reducing potential crashes.
> - Added informative print statements to provide better feedback during processing.
>
> #### 5. Optimized Data Retrieval and Output:
> - Modified the code to use a more modular approach for retrieving, processing, and saving data.
> - Enhanced the output by including additional context and relevant URLs, improving the readability and usefulness of responses.

---
> ### 09/06/2024
> 
> **Updated Files:**
> 
> - `requirements.txt`
>   - Updated with new dependencies for the Python environment.
>  
> - Ferguson Directory
>    - Change Agent 23 May 2024
>    - KOMO News 14 July 2024
>    - Race Announcment July 2024
>    - Seattle Times 9 June 2024
>    - Oxford Union 4 June 2024
>    - Ferguson Issues Official Website 
> - Reichert Directory
>    - Seattle Times 7 July 2024
>    - Race For Governor 23 July 2024
>    - KOMO News 20 May 2024
>    - Time For Change 5 April 2024
>    - Trump Teriffs 31 May 2024
>    - Reichert Issues Official Website
>       
> **Deleted Files:**
> 
> - `generated-client.zip`
>   - Removed unnecessary file to keep the repository clean.
> - `vectorized_chunks.pkl`
>   - Removed old vectorization data to reduce clutter.
> 
> **Added Files:**
> 
> - `entrypoint.sh`
>   - This script ensures that `OpenAPI__process_sources.py` runs first to completion before starting `debate_bot.py`.
> - `Dockerfile`
>   - Added to provide a containerized environment for the app, allowing it to be run locally via Docker.
> - `.gitattributes`  Enforce Unix-style line endings for all shell scripts in the repository
> 
> **Purpose:**
> 
> To enable a streamlined setup for running the Python Q&A app in a Dockerized environment, ensuring all dependencies are managed and the correct sequence of operations is executed.
> 
> **Key Features:**
> 
> - `entrypoint.sh` orchestrates the execution order, ensuring that all source processing is done before the chatbot starts.
> - Dockerized setup simplifies environment configuration and dependency management, making it easier for others to clone, build, and run the application locally.
> 
> **How to Build and Run:**
> 
> 1. Clone the repository:
> 
>    ```bash
>    git clone https://github.com/Human-Rights-First-Innovation-Lab/debate_bot
>    cd debate_bot
>    ```
> if it is located in an alternate branch use after cloning
>    ```bash
>    git checkout branch-name
>    ```
> 2. Ensure `.env` is placed in the same directory.
>    
> 3. Download the appropriate docker desktop from https://www.docker.com/products/docker-desktop/ for Windows navigate to https://docs.docker.com/desktop/install/windows-install/
>    
> 4. Build the Docker image:
> 
>    ```bash
>    docker build -t debate-bot .
>    ```
> 
> 5. Run the Docker container interactively:
>  Replace port mapping
>    ```bash
>    docker run -d -p 8080:8080 --env-file .env -it debate-bot
>    ```
> 
---
> ## 09/08/2024
>
> #### Add candidate win counts:
> - Added the `leaderboard_stats.py` file to hold functions related to getting stats for the leaderboard page
> - Added `get_winner_counts()` to get count of each candidate's wins

> - added a handler `stats_handler()` in `main.py` to provide the api the json structure with placeholder values
>

> ## 09/18/2024
>
> #### Add Response Flag and Session IDs:
> - Added Import Secrets
> - Added Flag Response if both candidates cannot provide an answer
> - Added  `router.get("/start-session/")`
>
>   JSON Payload
>   response body 
>   `{
>      "messages": "Session Started"
>      "session_id": "generated_session_token_here"
>   }`
>
>   response header
>   `Set-Cookie: session_id=generated_session_token_here; Path=/; HttpOnly`
>      
---
> ## 09/19/2024
>
> #### Update Pickle and Source Files
> - Updated app/data/embeddings/Ferguson.pkl
> - Updated app/data/embeddings/Reichert.pkl
> - Added into app/sources/ferguson - Ferguson_Second_Debate_Sept_18_2024.txt
> - Added into app/sources/reichert - Reichert_Second_Debate_Sept_18_2024.txt
> - added a handler `stats_handler()` in `main.py` to provide the api the json structure with placeholder values  

---
> ## 09/17/2024
>
> #### Add categorization:
> - Added the `categorize_question()` to generate categories for questions before saving questions into the db
> - Added `categorize_all_questions()` to update all rows with null category, in case of needing a bulk run 

> #### Add leaderboard stat functions and endpoint:
> - added an endpoint `/stats` to provide the api the json with all the leaderboard stats
> - includes wins, party affiliations, gender, age, and top categories




## 📜 License

This project is licensed under the **Testing Purposes License**. For more details, please see the [LICENSE](./LICENSE) file.


## 📚 Text Data Sources

Here are the key documents and speeches used as text data sources for our project. Each link provides access to publicly available content for analysis.

### 🗣️ Speeches and Public Statements can all be found in the app/sources directory


