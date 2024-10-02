import os
import pickle
import re
import csv
import json
from jose import jwt, JWTError
import numpy as np
from scipy.spatial.distance import cosine
import tiktoken  # OpenAI's tokenizer
import pandas as pd
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from datasets import Dataset 
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import openai
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, update, bindparam
from datetime import datetime
from openai import OpenAI


# Correctly load the .env file
dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path)

# Get API key from environment and debug print to check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print("Loaded API Key:", api_key)  # Debugging print statement

client = openai.OpenAI()

if api_key:
    # Initialize OpenAI with the loaded API key
    openai.api_key = api_key
else:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")
    
# Initialize OpenAI API key
#openai.api_key = os.getenv('OPENAI_API_KEY')

MYSQL_DATABASE = 'debatebot_prod'

# Database Configuration
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = os.getenv('MYSQL_PORT')
#MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

# Model to use for chat completions
model = 'gpt-4o-mini'

# Debugging: Print environment variable values
print(f"MySQL User: {MYSQL_USER}")
print(f"MySQL Host: {MYSQL_HOST}")
print(f"MySQL Port: {MYSQL_PORT}")
print(f"MySQL Database: {MYSQL_DATABASE}")

# SQLAlchemy engine
engine = sqlalchemy.create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')


async def validate_token(token: str):
    try:
        # Extract header from JWT to get the 'kid'
        unverified_header = jwt.get_unverified_header(token)

        # Fetch the JWKS (JSON Web Key Set) and find the correct key
        jwks = get_auth0_jwks()
        rsa_key = {}
        for key in jwks['keys']:
            if key['kid'] == unverified_header['kid']:
                rsa_key = {
                    'kty': key['kty'],
                    'kid': key['kid'],
                    'use': key['use'],
                    'n': key['n'],
                    'e': key['e']
                }
        if rsa_key:
            # Validate the JWT using Auth0 public key
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=AUTH0_M2M_AUDIENCE,
                issuer=f"https://{AUTH0_BASE_URL}/"
            )
            return payload
        else:
            raise HTTPException(status_code=401, detail="Unable to find the appropriate key.")
    except JWTError as e:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


# Function to create a database connection using MySQL Connector
def get_database_connection():
    try:
        print(f"Connecting to database: {MYSQL_DATABASE}")  # Debugging statement
        connection = mysql.connector.connect(
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            database=MYSQL_DATABASE
        )
        return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        raise e


# Function to create a database connection using SQLAlchemy
def get_database_engine():
    try:
        engine = create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}')
        return engine
    except Exception as e:
        print(f"Error creating SQLAlchemy engine: {e}")
        raise e

def insert_into_database(sql_string, vals):
    print(f"Executing SQL: {sql_string} with values {vals}")  # Debugging
    try:
        connection = get_database_connection()
        cursor = connection.cursor()

        # Ensure correct number of parameters for the SQL insert
        if 'INSERT INTO Query' in sql_string:
            # Make sure vals has the exact number of parameters that matches the SQL
            if len(vals) == 2:
                vals = (vals[0], vals[1], datetime.now())  # Add current timestamp

        cursor.execute(sql_string, vals)
        connection.commit()
        print(cursor.rowcount, "record inserted.")
        
    except Error as e:
        print(f"Error executing insert: {e}")
        raise e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def select_from_database(sql_string):
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        cursor.execute(sql_string)
        records = cursor.fetchall()
        return records
    except Error as e:
        print(f"Error executing select: {e}")
        raise e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to estimate the token length of text using OpenAI's tokenizer
def estimate_tokens(text, model_name=model):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# Function to truncate text to fit a specified number of tokens
def truncate_text_to_fit(text, max_tokens):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# Function to generate a response using the ChatGPT model
def generate_response(query, retrieved_texts, max_tokens=4096):
    context = ', '.join(retrieved_texts)
    system_message = {
        "role": "system",
        "content": "Answer the question as if you are the speaker. Answer with facts based only on the provided context, and if the context does not have enough relevant information, say that you do not have enough information to answer the question. Avoid any mention of specific political parties or candidates. Keep the answer short, clear, and concise."
    }
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    # Adjust the prompt if it exceeds the maximum token length
    if estimate_tokens(prompt) > max_tokens:
        context_length = max_tokens - estimate_tokens(f"Question: {query}\n\nAnswer:")
        truncated_retrieved_text = truncate_text_to_fit(context, context_length)
        prompt = f"Context: {truncated_retrieved_text}\n\nQuestion: {query}\n\nAnswer:"

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                system_message,
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.1
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text
    except Exception as e:
        print(f"Error generating chat completion: {e}")
        return None

# Function to save data to a CSV file
def save_to_csv(data, filename='chatbot_data.tsv'):
    file_exists = os.path.isfile(filename)
    # Open the file with UTF-8 encoding
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')  # Use tab as the delimiter
        if not file_exists:
            writer.writerow(['Query', 'Retrieved Text', 'Response', 'Filename'])  # Proper headers
        writer.writerow([data['query'], data['retrieved_text'], data['response'], data['filenames']])

# Function to save data to the database
def save_to_db(data):
    contexts = json.dumps(data['retrieved_text'])
    filenames = json.dumps(data['filenames'])
    vals = (data['query_id'], data['candidate_id'], data['response'], contexts, filenames, 0, float(data['answer_relevancy']), float(data['faithfulness'])) #TODO fill in vote
    insert_into_database(f"INSERT INTO Response (queryId, candidateId, response, contexts, filenames, userVoted, answerRelevancyScore, faithfulnessScore) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", vals)

# Function to generate OpenAI embeddings for a given text
def get_openai_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# Function to extract URL from the top of a text file
def extract_url_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the first line where the URL is expected
            url = file.readline().strip()
            # Validate if it is a URL
            if re.match(r'(https?://\S+)', url):
                return url
            else:
                print(f"No valid URL found at the top of {file_path}.")
                return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading URL from {file_path}: {e}")
        return None

# Function to find the best matching texts based on cosine similarity
def find_best_texts(query_embedding, pkl_filenames, txt_folder_path, n):
    best_retrieved_texts = []
    best_filenames = []
    best_similarities = []
    best_urls = []  # List to store URLs

    # Dictionary to cache URLs extracted from .txt files
    url_cache = {}

    # Process each .pkl file and retrieve texts
    for pkl_filename in pkl_filenames:
        with open(pkl_filename, 'rb') as file:
            vectorized_chunks = pickle.load(file)

            # Process each chunk in the .pkl file
            for embedding, chunk, chunk_filenames, chunk_urls in vectorized_chunks:
                # Extract the corresponding .txt filename
                txt_filename = chunk_filenames[0]  # Assuming chunk_filenames contains the original .txt filename
                
                # Compute similarity and store the best results
                similarity_score = cosine(query_embedding, embedding)
                if similarity_score > 0:
                    best_similarities.append(similarity_score)
                    best_retrieved_texts.append(chunk[0])
                    best_filenames.append(chunk_filenames[0])
                    best_urls.append(chunk_urls[0])  # Use the URL from the .pkl file

    # Combine results into a DataFrame and sort by similarity
    text_similarities = pd.DataFrame(
        {
            'texts': best_retrieved_texts,
            'filenames': best_filenames,
            'similarities': best_similarities,
            'urls': best_urls  # Include URLs in the DataFrame
        }
    )

    result = text_similarities.sort_values('similarities', ascending=True)

    # Print the top results for debugging
    print(result.head())

    # Return the top 'n' results
    return result.head(n)

# Function to compute scoring metrics
def get_scoring_metrics(query, response, contexts):
    data = {
        'question': [query],
        'answer': [response],
        'contexts' : [contexts]
    }
    dataset = Dataset.from_dict(data)
    score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    score.to_pandas()
    return score

def get_sqlachemy_connection():
    """
    This gets a connection to the HRF debate bot mysql database using SQLAlchemy
    Args:
        none

    Returns:
        sqlalchemy connection object

    """
    connection = engine.connect()
    return connection

def get_winner_percents():
    """
    This gets the percent of wins each candidate had. One win is one session where the
    user selected this candidate a majority of the time. Ties do not count as a win for anyone.

    Args:
        none

    Returns:
        dict {"ferguson_win_pct": float, "reichert_win_pct": float}

    """
    connection = get_sqlachemy_connection()
    query = '''
    select r.userVoted,
        r.candidateId,
        q.sessionId
    from Response r join Query q on r.queryId = q.id 
    '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    votes_per_candidate_per_session = df.groupby(['sessionId', 'candidateId']).sum('userVoted').reset_index().pivot(index='sessionId', columns='candidateId', values='userVoted')
    print(votes_per_candidate_per_session)
    winner_per_session = votes_per_candidate_per_session
    winner_per_session['ferguson_winner'] = np.where(winner_per_session[1]>winner_per_session[2], 1, 0)
    winner_per_session['reichert_winner'] = np.where(winner_per_session[2]>winner_per_session[1], 1, 0)
    ferguson_win_count = winner_per_session['ferguson_winner'].sum()
    print("ferguson wins:", ferguson_win_count)
    reichert_win_count = winner_per_session['reichert_winner'].sum()
    print("reichert wins:", reichert_win_count)
    total_count = ferguson_win_count + reichert_win_count
    result = {"ferguson_win_pct": float(ferguson_win_count/total_count), "reichert_win_pct": float(reichert_win_count/total_count)}
    return result

def categorize_question(question):
    prompts=[{"role": "system", "content": 
        """
        You are a political scientist at a national newspaper. 
        Categorize the next statements to their most relevant political key issue.
        If the statement is not similar to any key issue, then categorize it as Other.
        The key issues are:
        Economy,
        Healthcare,
        Education,
        Immigration,
        Environment,
        National Security,
        Criminal Justice,
        Social Justice,
        Tax Policy,
        Gun Control,
        Infrastructure,
        Public Safety,
        Foreign Policy,
        Housing,
        Social Welfare Programs,
        Drug Policy,
        Veterans Affairs,
        Technology and Privacy,
        Election Integrity,
        Reproductive Rights,
        Gender,
        Religious Freedom
        
        """
        }]
    prompts.append({"role": "user", "content": question})
    answer = openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=20,
                messages=prompts,
                temperature=0
            )
    
    category = answer.choices[0].message.content.strip()
    return category

def categorize_all_questions():
    connection = get_sqlachemy_connection()

    # get the questions from the db which have not been categorized yet
    query = '''
    select id as query_id, query
    from Query
    where category is null
    '''
    questions = pd.read_sql_query(query, connection)

    # categorize the questions
    questions['category'] = questions['query'].apply(categorize_question)
    df_dict = questions.to_dict('records')

    # update those rows with their category
    metadata = sqlalchemy.MetaData()
    query_table = sqlalchemy.Table('Query', metadata,
        Column("id", Integer, primary_key=True),
        Column("query", String),
        Column("category", String),
    )
    stmt = (
        update(query_table)
        .where(query_table.c.id == bindparam("query_id"))
        .values(category=bindparam("category"))
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            df_dict,
        )
    print(stmt)

def get_top_categories(n):
    """
    This gets the names and counts of the top n most asked categories of questions.

    Args:
        n (int): the number of categories to get

    Returns:
        list of tuples, e.g. [("Economy", 9),
                                ("Healthcare", 8),
                                ("Education", 7),
                                ("Immigration", 6)]
    """
    connection = get_sqlachemy_connection()
    query = '''
    select category, count(1) as n
    from Query
    group by category
    order by count(1) desc
    '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    top_n = df.head(n)
    top_categories = list(top_n.itertuples(index=False, name=None))
    print(top_categories)
    return top_categories

# participant demographics, % party 
def get_participant_parties():
    """
    This gets the parties and percent of participants who said they were affiliated with each.

    Args:
        none

    Returns:
        dict, {"republican": int, "democrat": int, "other": int, "independent": int, "prefer_not_to_say": int}
    """
    connection = get_sqlachemy_connection()
    query = '''
        select partyName, count(1) as n 
        from Session s join Party p on s.partyId = p.id
        group by partyName
        '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    total_count = df.n.sum()
    republican_count = df.loc[df['partyName'] == "Republican"]
    republican_count = republican_count.n.sum()
    republican_percent = float(republican_count/total_count)

    democrat_count = df.loc[df['partyName'] == "Democrat"]
    democrat_count = democrat_count.n.sum()
    democrat_percent = float(democrat_count/total_count)

    independent_count = df.loc[df['partyName'] == "Independent"]
    independent_count = independent_count.n.sum()
    independent_percent = float(independent_count/total_count)

    other_count = df.loc[df['partyName'] == "Other"]
    other_count = other_count.n.sum()
    other_percent = float(other_count/total_count)

    nosay_count = df.loc[df['partyName'] == "Prefer Not To Say"]
    nosay_count = nosay_count.n.sum()
    nosay_percent = float(nosay_count/total_count)

    result = {"republican": republican_percent,
        "democrat": democrat_percent,
        "other": other_percent,
        "independent": independent_percent,
        "prefer_not_to_say": nosay_percent}
    print(result)
    return result

# % by gender
def get_participant_genders():
    """
    This gets the genders and percent of participants who said they identify with each.

    Args:
        none

    Returns:
        dict, {"male": int, "female": int, "nonbinary": int, "prefer_not_to_say": int}
    """
    connection = get_sqlachemy_connection()
    query = '''
        select genderIdentity, count(1) as n 
        from Session
        group by genderIdentity
        '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    total_count = df.n.sum()
    male_count = df.loc[df['genderIdentity'] == "male"]
    male_count = male_count.n.sum()
    male_percent = float(male_count/total_count)

    female_count = df.loc[df['genderIdentity'] == "female"]
    female_count = female_count.n.sum()
    female_percent = float(female_count/total_count)

    nonbinary_count = df.loc[df['genderIdentity'] == "non_binary"]
    nonbinary_count = nonbinary_count.n.sum()
    nonbinary_percent = float(nonbinary_count/total_count)

    nosay_count = df.loc[df['genderIdentity'] == "prefer_not_to_say"]
    nosay_count = nosay_count.n.sum()
    nosay_percent = float(nosay_count/total_count)

    result = {"male": male_percent,
        "female": female_percent,
        "nonbinary": nonbinary_percent,
        "prefer_not_to_say": nosay_percent}
    print(result)
    return result

# % by age range
def get_participant_ages():
    """
    This gets the age brackets and percent of participants who said they identify with each.

    Args:
        none

    Returns:
        dict, {"<18": float, "18-35": float, "36-55": float, "56-75": float, "76+": float}
        
    """
    connection = get_sqlachemy_connection()
    query = '''
        select case 
            when age <18 then '<18'
            when age >18 and age <35 then '18-35'
            when age >35 and age <55 then '36-55'
            when age >55 and age <70 then '56-75'
            when age >75 then '76+'
            end
            as age_bucket, 1 as n 
        from Session
        '''
    df = pd.read_sql_query(query, connection)
    connection.close()
    total_count = df.n.sum()
    count_0_18 = df.loc[df['age_bucket'] == "<18"]
    count_0_18 = count_0_18.n.sum()
    pct_0_18 = float(count_0_18/total_count)

    count_18_35 = df.loc[df['age_bucket'] == "18-35"]
    count_18_35 = count_18_35.n.sum()
    pct_18_35 = float(count_18_35/total_count)

    count_36_55 = df.loc[df['age_bucket'] == "36-55"]
    count_36_55 = count_36_55.n.sum()
    pct_36_55 = float(count_36_55/total_count)

    count_56_75 = df.loc[df['age_bucket'] == "56-75"]
    count_56_75 = count_56_75.n.sum()
    pct_56_75 = float(count_56_75/total_count)

    count_76 = df.loc[df['age_bucket'] == "76+"]
    count_76 = count_76.n.sum()
    pct_76 = float(count_76/total_count)

    result = {"pct_0_18": pct_0_18,
        "pct_18_35": pct_18_35,
        "pct_36_55": pct_36_55,
        "pct_56_75": pct_56_75,
        "pct_76": pct_76}
    print(result)
    return result
