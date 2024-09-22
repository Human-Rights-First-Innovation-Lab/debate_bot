from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from app.endpoints import router  # Ensure this import points to your endpoints file
import os
import openai
from app.utils import (
    insert_into_database,
    select_from_database,
    generate_response,
    get_openai_embedding,
    find_best_texts,
    save_to_db,
)
from dotenv import load_dotenv


# Access the API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dbapi.hrfinnovation.org",  # Main Branch FE
        "https://dbapi-stag.hrfinnovation.org",  # Dev branch FE
        "https://debatebot-client.vercel.app",
        "https://debatebot-client.vercel.app/",
        "https://debatebot-client-git-develop-hrf-innovation-lab.vercel.app/",
        "https://debatebot-client-git-develop-hrf-innovation-lab.vercel.app"
        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to redirect HTTP to HTTPS if necessary
@app.middleware("http")
async def enforce_https(request: Request, call_next):
    # Check if the request was forwarded via HTTPS
    if request.url.scheme == "http" and request.headers.get("x-forwarded-proto") != "https":
        # Redirect to HTTPS version of the URL
        https_url = request.url.replace(scheme="https")
        return RedirectResponse(url=https_url)

    response = await call_next(request)
    return response
    
# Include the router from endpoints
app.include_router(router)

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Debate Bot API!"}
