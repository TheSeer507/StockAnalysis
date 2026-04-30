# Configuration — values are loaded from environment variables.
# Copy .env.example to .env and fill in your real credentials.
# NEVER commit .env to version control.

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY        = os.getenv("CAPITAL_API_KEY", "")
API_SECRET     = os.getenv("CAPITAL_API_SECRET", "")
LOGIN          = os.getenv("CAPITAL_LOGIN", "")
PASSWORD       = os.getenv("CAPITAL_PASSWORD", "")
CUSTOMPASSWORD = os.getenv("CAPITAL_CUSTOM_PASSWORD", "")

PRICE_CEILING = 50.0

# Set to True if you are using the demo API, False for the live API
IS_DEMO = os.getenv("CAPITAL_IS_DEMO", "true").lower() == "true"

DEMO_API_URL = "https://demo-api-capital.backend-capital.com"
PROD_API_URL = "https://api-capital.backend-capital.com"

# Decide which one to use
API_URL = DEMO_API_URL if IS_DEMO else PROD_API_URL

'''
Base URL: https://api-capital.backend-capital.com/
Base demo URL: https://demo-api-capital.backend-capital.com/
'''