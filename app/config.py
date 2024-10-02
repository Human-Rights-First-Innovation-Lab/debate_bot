from pydantic import BaseSettings
from dotenv import load_dotenv
import os
import openai
# Load environment variables from the .env file located in the /app directory
#load_dotenv(dotenv_path="/app/.env")

# Access the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI client
openai.api_key = openai_api_key

MYSQL_DATABASE = 'debatebot_prod'

class Settings(BaseSettings):
    # App settings
    app_name: str = "Debate Bot"
    debug: bool = True

    # Database settings
    mysql_user: str = os.getenv("MYSQL_USER")
    mysql_password: str = os.getenv("MYSQL_PASSWORD")
    mysql_host: str = os.getenv("MYSQL_HOST")
    mysql_port: int = int(os.getenv("MYSQL_PORT", 3306))

    mysql_database: str = "debatebot_prod"


    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    
    # Auth0 settings
    auth0_domain: str = os.getenv("AUTH0_DOMAIN", "https://hrf-production.us.auth0.com/")
    auth0_client_id: str = os.getenv("AUTH0_CLIENT_ID")
    auth0_client_secret: str = os.getenv("AUTH0_CLIENT_SECRET")
    auth0_audience: str = os.getenv("AUTH0_AUDIENCE", "https://dbapi.hrfinnovation.org/api/v2/")

    class Config:
        env_file = ".env"

# Instantiate the settings
settings = Settings()

