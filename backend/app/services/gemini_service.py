import requests
import json
import os
import logging
from app.core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class GeminiChatbot:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        self.transcript_content = None
        logger.info("Initialized GeminiChatbot with API key: %s", self.api_key[:10] + "...")

    def read_transcript(self, file_path: str) -> bool:
        """Read the transcript file and store its contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.transcript_content = file.read()
            logger.info("Successfully read transcript file: %s", file_path)
            return True
        except Exception as e:
            logger.error("Error reading transcript file: %s", str(e))
            return False

    def send_message(self, query: str) -> str:
        """Send a message to Gemini API and get response"""
        try:
            if not self.transcript_content:
                logger.error("No transcript content available")
                return "Error: No transcript content available"

            # Create prompt with transcript context
            prompt = f"""Based on this video transcript:

{self.transcript_content}

Question: {query}

Please provide a detailed answer based only on the information in the transcript."""

            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }

            headers = {
                "Content-Type": "application/json"
            }

            logger.info("Sending request to Gemini API")
            response = requests.post(self.api_url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Received successful response from Gemini API")
                if 'candidates' in result:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.error("Unexpected response format: %s", json.dumps(result))
                    return "Error: Unexpected response format from API"
            else:
                logger.error("API error: %s - %s", response.status_code, response.text)
                return f"Error: API returned status code {response.status_code} - {response.text}"

        except Exception as e:
            logger.error("Error in send_message: %s", str(e))
            return f"Error sending message: {str(e)}"

    def generate_title(self, transcript_text):
        """Generate a very concise title (4-5 words) for the video"""
        try:
            prompt = """Based on this transcript, generate an extremely concise title (maximum 4-5 words) that captures the main topic. 
            Make it clear and specific, like a headline.
            Transcript:
            
            """ + transcript_text

            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }

            headers = {
                "Content-Type": "application/json"
            }

            logger.info("Sending title generation request to Gemini API")
            response = requests.post(self.api_url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Received successful response for title generation")
                if 'candidates' in result:
                    title = result['candidates'][0]['content']['parts'][0]['text']
                    # Clean and limit the title
                    title = title.strip().strip('"').strip()
                    words = title.split()[:5]  # Limit to 5 words
                    return ' '.join(words)
                else:
                    logger.error("Unexpected response format for title: %s", json.dumps(result))
            
            logger.error("Title generation failed: %s - %s", response.status_code, response.text)
            return "Untitled Video"
        except Exception as e:
            logger.error("Error generating title: %s", str(e))
            return "Untitled Video"