import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, ValidationError
import sys
import os
import re
from pathlib import Path
from app.services.video_service import VideoService
from app.services.gemini_service import GeminiChatbot
import shutil
from typing import Literal
from app.core.config import GEMINI_API_KEY, API_CORS_ORIGINS, STORAGE_DIR


# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize VideoService instance
video_service = VideoService()
gemini_chatbot = GeminiChatbot()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track processing status
video_processing_status = {}

# At the start of the app
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)
    logger.info(f"Created {STORAGE_DIR} directory")

class VideoRequest(BaseModel):
    url: HttpUrl
    mode: Literal["normal", "advanced"] = "normal"

class ChatRequest(BaseModel):
    message: str
    videoId: str

class TitleUpdateRequest(BaseModel):
    videoId: str
    newTitle: str

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"message": "Invalid input", "details": exc.errors()},
    )

@app.post("/process-video")
async def process_video(request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        video_id = video_service.get_video_id(str(request.url))
        
        # Add mode to processing status
        video_processing_status[video_id] = {
            "status": "processing",
            "mode": request.mode
        }
        
        # Process video in background
        background_tasks.add_task(
            process_video_task, 
            str(request.url), 
            video_id,
            request.mode
        )
        
        return {"video_id": video_id}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_task(url: str, video_id: str, mode: str):
    try:
        # Pass mode to process_video
        result = video_service.process_video(url, mode)
        video_processing_status[video_id] = {
            "status": "completed",
            "mode": mode
        }
        logger.info(f"Video processing completed for {video_id} in {mode} mode")
    except Exception as e:
        logger.error(f"Error in background video processing: {str(e)}")
        video_processing_status[video_id] = {
            "status": "failed",
            "mode": mode
        }

@app.get("/video-status/{video_id}")
async def get_video_status(video_id: str):
    status = video_processing_status.get(video_id, {"status": "not_found", "mode": "normal"})
    return status  # Return the entire status object, not just the string

@app.post("/start-chat")
async def start_chat(request: ChatRequest):
    try:
        video_id = request.videoId
        logger.info(f"Starting chat for video {video_id}")
        
        # Check if video processing is complete
        video_status = video_processing_status.get(video_id)
        logger.info(f"Video status: {video_status}")
        
        if not video_status:
            raise HTTPException(status_code=400, detail="Video not found")
            
        if video_status.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Video processing not completed")

        # Construct path to transcript file using video ID
        transcript_path = os.path.join(STORAGE_DIR, video_id, "video_transcript.txt")
        logger.info(f"Looking for transcript at: {transcript_path}")
        
        if not os.path.exists(transcript_path):
            logger.error(f"Transcript not found at {transcript_path}")
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        # Read transcript
        logger.info("Reading transcript...")
        if not gemini_chatbot.read_transcript(transcript_path):
            logger.error("Failed to read transcript")
            raise HTTPException(status_code=500, detail="Failed to read transcript")
        
        # Get response from Gemini
        logger.info(f"Getting response for message: {request.message}")
        response = gemini_chatbot.send_message(request.message)
        
        return {
            "message": response,
            "mode": video_status.get("mode", "normal")
        }
        
    except Exception as e:
        logger.error(f"Error in chat process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/video-history")
async def get_video_history():
    try:
        # Ensure video_findings directory exists
        if not os.path.exists(STORAGE_DIR):
            return []
        
        # Get all subdirectories in video_findings
        video_dirs = [d for d in os.listdir(STORAGE_DIR) 
                     if os.path.isdir(os.path.join(STORAGE_DIR, d))]
        
        history = []
        for video_id in video_dirs:
            try:
                video_dir = os.path.join(STORAGE_DIR, video_id)
                
                # Check if this is a complete video directory
                transcript_path = os.path.join(video_dir, "video_transcript.txt")
                title_path = os.path.join(video_dir, "title.txt")
                
                if os.path.exists(transcript_path):
                    # Get title from title.txt or generate from transcript
                    title = "Untitled Video"
                    if os.path.exists(title_path):
                        with open(title_path, 'r', encoding='utf-8') as f:
                            title = f.read().strip()
                    else:
                        # If no title file, read first few lines of transcript
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript_preview = f.read(1000)  # Read first 1000 chars
                            title = transcript_preview.split('\n')[0][:50]
                    
                    # Get the directory creation time as timestamp
                    timestamp = os.path.getctime(video_dir)
                    
                    # Determine mode from directory structure or default to normal
                    mode = "advanced" if os.path.exists(os.path.join(video_dir, "frames")) else "normal"
                    
                    history.append({
                        "id": video_id,
                        "mode": mode,
                        "timestamp": timestamp * 1000,  # Convert to milliseconds for JS
                        "title": title
                    })
            except Exception as e:
                logger.warning(f"Error processing history item {video_id}: {str(e)}")
                continue
        
        # Sort by timestamp, newest first
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return history
    except Exception as e:
        logger.error(f"Error getting video history: {str(e)}")
        return []  # Return empty list instead of error for better UX

@app.post("/load-historical-video/{video_id}")
async def load_historical_video(video_id: str):
    try:
        video_dir = os.path.join(STORAGE_DIR, video_id)
        if not os.path.exists(video_dir):
            raise HTTPException(status_code=404, detail="Video not found")

        # Determine mode from directory structure
        mode = "advanced" if os.path.exists(os.path.join(video_dir, "frames")) else "normal"
        
        # Update video processing status
        video_processing_status[video_id] = {
            "status": "completed",
            "mode": mode
        }
        
        return {"status": "success", "mode": mode}
    except Exception as e:
        logger.error(f"Error loading historical video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-title")
async def update_title(request: TitleUpdateRequest):
    try:
        logger.info(f"Attempting to update title for video {request.videoId} to: {request.newTitle}")
        
        video_dir = os.path.join(STORAGE_DIR, request.videoId)
        logger.info(f"Looking for video directory at: {video_dir}")
        
        if not os.path.exists(video_dir):
            logger.error(f"Video directory not found: {video_dir}")
            raise HTTPException(status_code=404, detail=f"Video directory not found: {video_dir}")

        title_path = os.path.join(video_dir, "title.txt")
        logger.info(f"Will save title to: {title_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(title_path), exist_ok=True)
        
        # Update the title
        with open(title_path, 'w', encoding='utf-8') as f:
            f.write(request.newTitle)
        
        # Update the status if it exists
        if request.videoId in video_processing_status:
            video_processing_status[request.videoId]["title"] = request.newTitle
            logger.info("Updated title in processing status")
        
        logger.info(f"Successfully updated title for video {request.videoId}")
        return {"status": "success", "title": request.newTitle}
    except Exception as e:
        logger.error(f"Error updating title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-title/{video_id}")
async def get_video_title(video_id: str):
    try:
        video_dir = os.path.join(STORAGE_DIR, video_id)
        title_path = os.path.join(video_dir, "title.txt")
        
        if os.path.exists(title_path):
            with open(title_path, 'r', encoding='utf-8') as f:
                title = f.read().strip()
            return {"title": title}
        else:
            raise HTTPException(status_code=404, detail="Title not found")
    except Exception as e:
        logger.error(f"Error getting video title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)