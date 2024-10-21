import requests
import logging
import traceback
from typing import Optional, Dict, List, Tuple, Union
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from jose import jwt, JWTError
from pydantic import BaseModel
import os
from datetime import datetime
import zlib
from app.utils import (
    insert_into_database,
    select_from_database,
    generate_response,
    categorize_question,
    get_openai_embedding,
    find_best_texts,
    save_to_db,
    get_participant_parties,
    get_participant_genders,
    get_participant_ages,
    get_top_categories,
    get_winner_percents,
    validate_token,
    embed_timestamp_in_url
)
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_PAYLOAD_SIZE = 256000  # 250KB

def validate_payload_size(payload):
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError("Payload size exceeds the maximum allowed limit of 250KB")

def validate_decompressed_payload_size(compressed_payload):
    decompressed_payload = zlib.decompress(compressed_payload)
    if len(decompressed_payload) > MAX_PAYLOAD_SIZE:
        raise ValueError("Decompressed payload size exceeds the maximum allowed limit of 250KB")
    return decompressed_payload

# Initialize FastAPI router
router = APIRouter()

# Pydantic models for request and response validation
class QueryRequest(BaseModel):
    query: str
    session_id: str

class StartSessionResponseModel(BaseModel):
    message: Optional[str]
    session_id: str

class ResponseModel(BaseModel):
    query_id: int
    session_id: Optional[str]  # Make this optional if it's not always present
    responses: Dict[str, Dict[str, str]]

class StatsResponseModel(BaseModel):
    candidate_wins: Dict [str, float]
    participant_party: Dict[str, Union[int, float]]
    participant_age: Dict[str, Union[int, float]]
    participant_gender: Dict[str, Union[int, float]]
    top_categories: Union[Dict[str, int], List[Tuple[Union[str, None], int]]] 

class SaveRequest(BaseModel):
    query_id: int
    candidate_id: int
    response: str
    retrieved_text: List[str]
    filenames: List[str]
    user_voted: int
    contexts: List[str]
    answer_relevancy: float
    faithfulness: float

# Load Auth0 settings from environment DO NOT PUT https:// IN AUTH0 DOMAIN
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "hrf-alt-dev.us.auth0.com")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://dbapi-stag.hrfinnovation.org/api/v2/")
ALGORITHMS = ["RS256"]

if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
    raise ValueError("Missing Auth0 environment variables. Ensure they are set properly.")

# Fetch Auth0 JWKS for verifying RS256 tokens
jwks_cache = None

def get_jwks():
    global jwks_cache
    if jwks_cache is None:
        url = f'https://{AUTH0_DOMAIN}/.well-known/jwks.json'
        response = requests.get(url)
        if response.status_code == 200:
            jwks_cache = response.json()
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch JWKS")
    return jwks_cache

# Extract the signing key from JWKS
def get_signing_key(token: str):
    try:
        unverified_header = jwt.get_unverified_header(token)
        jwks = get_jwks()

        # Search for the correct key in the JWKS based on 'kid'
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                return rsa_key
        raise HTTPException(status_code=403, detail="Unable to find appropriate key")
    except JWTError as e:
        raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")
    except KeyError:
        raise HTTPException(status_code=403, detail="Missing 'kid' in JWT header")

# JWT token verification helper for RS256
def verify_rs256_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        logger.error("Authorization header missing")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    logger.info(f"Authorization header received: {auth_header}")
    
    try:
        token = auth_header.split("Bearer ")[1]
    except IndexError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format. Expected: Bearer <token>")

    rsa_key = get_signing_key(token)
    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")

# Start session endpoint
@router.get("/start-session/", response_model=StartSessionResponseModel)
async def start_session(request: Request, token_payload: dict = Depends(verify_rs256_token)):
    logger.info("Start session endpoint invoked")
    try:
        # Retrieve the 'sub' claim from the token, which represents the authenticated user
        session_token = token_payload.get('sub')

        # Check if the token contains a session identifier (sub claim)
        if not session_token:
            raise HTTPException(status_code=400, detail="Session ID is missing in the token")

        # Log the received session token (client ID)
        logger.info(f"Session started for user ID (sub): {session_token}")

        # Respond with a success message and the session ID
        return {"message": "Session started successfully", "session_id": session_token}

    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise HTTPException(status_code=403, detail="Invalid token or session")

    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while starting the session")

# Generate response endpoint
@router.post("/generate-response/", response_model=ResponseModel)
async def generate_response_endpoint(request: Request, req_body: QueryRequest, token_payload: dict = Depends(verify_rs256_token)):
    try:
        session_id = req_body.session_id
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is missing")
        query = req_body.query
        if not query:
            raise HTTPException(status_code=400, detail="Query is missing")
        
        logger.info(f"Authenticated user: {token_payload.get('sub')}")  
        
        # Categorize question
        category = categorize_question(query)

        # Insert user query into the database
        vals = (session_id, query, datetime.now(), category)
        insert_into_database("INSERT INTO Query (sessionId, query, timestamp, category) VALUES (%s, %s, %s, %s)", vals)

        # Retrieve the last query from the database
        query_id, query = select_from_database("SELECT id, query FROM Query ORDER BY id DESC LIMIT 1")[0]

        # Generate embedding for the user's query
        query_embedding = get_openai_embedding(query)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        ### **Retrieve and process texts for Reichert**
        # Retrieve texts and metadata for Reichert
        best_texts_df_reichert = find_best_texts(
            query_embedding,
            ['app/data/embeddings/vectorized_chunks_reichert.pkl'],
            'sources/reichert',
            4
        )

        # Process best_texts_df_reichert
        if not best_texts_df_reichert.empty:
            best_texts_df_reichert = best_texts_df_reichert.reset_index(drop=True)
            best_retrieved_texts_reichert = best_texts_df_reichert["texts"].tolist()
            source_url_reichert = embed_timestamp_in_url(
                best_texts_df_reichert.iloc[0]["urls"],
                best_texts_df_reichert.iloc[0]["timestamps"]
            )
        else:
            best_retrieved_texts_reichert = []
            source_url_reichert = "No URL found"

        # Generate a response for Reichert
        if best_retrieved_texts_reichert:
            best_response_reichert = generate_response(query, best_retrieved_texts_reichert)
        else:
            best_response_reichert = "This candidate has not spoken publicly on this topic, therefore we are unable to give a response at this time."

        ### **Retrieve and process texts for Ferguson**
        # Retrieve texts and metadata for Ferguson
        best_texts_df_ferguson = find_best_texts(
            query_embedding,
            ['app/data/embeddings/vectorized_chunks_ferguson.pkl'],
            'sources/ferguson',
            4
        )

        # Process best_texts_df_ferguson
        if not best_texts_df_ferguson.empty:
            best_texts_df_ferguson = best_texts_df_ferguson.reset_index(drop=True)
            best_retrieved_texts_ferguson = best_texts_df_ferguson["texts"].tolist()
            source_url_ferguson = embed_timestamp_in_url(
                best_texts_df_ferguson.iloc[0]["urls"],
                best_texts_df_ferguson.iloc[0]["timestamps"]
            )
        else:
            best_retrieved_texts_ferguson = []
            source_url_ferguson = "No URL found"

        # Generate a response for Ferguson
        if best_retrieved_texts_ferguson:
            best_response_ferguson = generate_response(query, best_retrieved_texts_ferguson)
        else:
            best_response_ferguson = "This candidate has not spoken publicly on this topic, therefore we are unable to give a response at this time."

        # Flag non-answers from candidates
        if "has not spoken publicly" in best_response_reichert.lower():
            source_url_reichert = "No URL found"
        if "has not spoken publicly" in best_response_ferguson.lower():
            source_url_ferguson = "No URL found"

        # Prepare the dictionary response
        response_data_dict = {
            "query_id": query_id,
            "session_id": session_id,
            "responses": {
                "reichert": {
                    "response": best_response_reichert,
                    "source_url": source_url_reichert,
                },
                "ferguson": {
                    "response": best_response_ferguson,
                    "source_url": source_url_ferguson,
                }
            }
        }

        # Log saving the response before performing the actual save operation
        logger.info(f"Saving response for query_id: {query_id}")

        # Save responses to the database
        save_to_db({
            "query_id": query_id,
            "candidate_id": 1,
            "response": best_response_reichert,
            "retrieved_text": best_retrieved_texts_reichert,
            "filenames": [txt for txt in best_texts_df_reichert["filenames"].tolist()],
            "user_voted": 0,
            "contexts": best_retrieved_texts_reichert,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0
        })

        save_to_db({
            "query_id": query_id,
            "candidate_id": 2,
            "response": best_response_ferguson,
            "retrieved_text": best_retrieved_texts_ferguson,
            "filenames": [txt for txt in best_texts_df_ferguson["filenames"].tolist()],
            "user_voted": 0,
            "contexts": best_retrieved_texts_ferguson,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0
        })

        return JSONResponse(content=response_data_dict)

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the response.")

# Stats endpoint
@router.get("/stats", response_model=StatsResponseModel)
async def stats_handler(token_payload: dict = Depends(verify_rs256_token)):
    return {
        "candidate_wins": get_winner_percents(),
        "participant_party": get_participant_parties(),
        "participant_age": get_participant_ages(),
        "participant_gender": get_participant_genders(),
        "top_categories": get_top_categories(10)
    }
