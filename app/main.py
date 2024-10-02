from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from authlib.integrations.starlette_client import OAuth
import requests
from app.endpoints import router

# Initialize FastAPI app
app = FastAPI()

app.include_router(router)

# Hardcoded Auth0 Configuration
AUTH0_BASE_URL = 'https://hrf-alt-dev.us.auth0.com'
AUTH0_M2M_AUDIENCE = 'https://dbapi-stag.hrfinnovation.org/api/v2/'
AUTH0_CLIENT_ID = 'WMgfTk583qBpRC9bWDohMH8VOyWo2m2W'  
AUTH0_CLIENT_SECRET = 'UmO8p22fzh_pciTMQXT-nCEZpZzdXy9_iteR2l9UYvPpNV1qvP1l1N-5jvpDpPWD'  

# Add trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["dbapi-stag.hrfinnovation.org", "*.dbapi-stag.hrfinnovation.org", "debatebot.hrfinnovation.org", "*.debatebot.hrfinnovation.org""localhost"],
)  

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dbapi.hrfinnovation.org",
        "https://dbapi-stag.hrfinnovation.org",
        "https://debatebot-client.vercel.app",
        "https://debatebot.hrfinnovation.org",
        "https://debatebot-client-git-develop-hrf-innovation-lab.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth
oauth = OAuth()

# Fetch Auth0 JWKS for verifying RS256 tokens
def get_auth0_jwks():
    url = f'{AUTH0_BASE_URL}/.well-known/jwks.json'
    jwks = requests.get(url).json()
    return jwks



# Protected route with token validation
@app.get("/secure-data")
async def get_secure_data(request: Request):
    token = request.headers.get('Authorization')
    if token:
        token = token.split("Bearer ")[-1]  # Remove "Bearer" from the token
    else:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # Validate the JWT token
    payload = await validate_token(token)
    
    # Return a secure response
    return {"message": "Secure data retrieved successfully", "user": payload}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Debate Bot API!"}
