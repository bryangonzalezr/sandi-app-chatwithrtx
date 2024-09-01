import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    _ngrok_url = os.getenv("NGROK_URL")

    @classmethod
    def update_ngrok_url(cls, new_url: str):
        cls._ngrok_url = new_url
        print(f"ngrok_url updated to: {cls._ngrok_url}")

    @classmethod
    def get_ngrok_url(cls):
        return cls._ngrok_url