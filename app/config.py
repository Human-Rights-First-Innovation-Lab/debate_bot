from pydantic import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from the .env file located in the /app directory
#load_dotenv(dotenv_path="/app/.env")

# Access the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI client
import openai
openai.api_key = openai_api_key

MYSQL_DATABASE = 'debatebot_prod'

class Settings(BaseSettings):
    # Application settings
    app_name: str = "FastAPI Application"
    debug: bool = True

    # Database settings
    mysql_user: str = os.getenv("MYSQL_USER")
    mysql_password: str = os.getenv("MYSQL_PASSWORD")
    mysql_host: str = os.getenv("MYSQL_HOST")
    mysql_port: int = int(os.getenv("MYSQL_PORT", 3306))
    #mysql_database: str = os.getenv("MYSQL_DATABASE")

    # OpenAI API settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # Other settings
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"

# Instantiate the settings
settings = Settings()

