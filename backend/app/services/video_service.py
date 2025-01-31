import yt_dlp
import os
import whisper
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from datetime import timedelta
import logging
import subprocess
import shutil
from pytube import YouTube
import io
from PIL import Image
import sys
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from .gemini_service import GeminiChatbot
from app.core.config import STORAGE_DIR, GEMINI_API_KEY

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoService:
    def __init__(self):
        self.base_output_dir = STORAGE_DIR
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.base_output_dir, 'video.%(ext)s'),
        }
        self.whisper_model = whisper.load_model("base")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.cnn_model = models.resnet18(pretrained=True)
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize DocTR with efficient model configuration
        logger.info("Initializing DocTR with optimized settings...")
        try:
            self.ocr_model = ocr_predictor(
                det_arch='db_resnet50',    # Efficient and accurate detection model
                reco_arch='crnn_vgg16_bn', # Best recognition model
                pretrained=True,
                assume_straight_pages=True, # Faster if text is mostly horizontal
                straighten_pages=True,
                detect_orientation=True,    # Detect and correct text orientation
                detect_language=True       # Auto language detection
            )
            logger.info("DocTR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocTR: {str(e)}")
            raise

        # Initialize gemini chatbot with key from config
        self.gemini_chatbot = GeminiChatbot()

    def extract_features(self, frame):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame).unsqueeze(0)
            frame = frame.to(self.device)
            with torch.no_grad():
                features = self.cnn_model(frame)
            return features.cpu().squeeze().numpy()
        except Exception as e:
            logger.error(f"Error in extract_features: {str(e)}")
            raise

    def get_video_output_dir(self, url):
        """Create and return a unique directory for each video"""
        video_id = self.get_video_id(url)
        video_dir = os.path.join(self.base_output_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)
        return video_dir

    def download_video(self, url):
        video_dir = self.get_video_output_dir(url)
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height>=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(video_dir, 'video.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            # Video quality settings
            'format_sort': [
                'res:1080',
                'fps:60',
                'codec:h264',
                'size',
                'br'
            ],
            # Force highest quality
            'prefer_free_formats': False,
            'merge_output_format': 'mp4',
            # Additional quality settings
            'writethumbnail': True,
            'writesubtitles': True,
            'subtitlesformat': 'srt',
            'keepvideo': True,
            'videoformat': 'mp4',
            # Force higher quality
            'format_sort_force': True,
            # Add quality preferences
            'video_format_sort': [
                'height:1080',
                'height:720',
                'fps',
                'codec:h264',
                'size',
                'br'
            ],
            # Force download of separate video and audio streams
            'format_sort_prefix': 'height:1080,fps:60,codec:h264'
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Starting video download with high quality settings...")
                info = ydl.extract_info(url, download=True)
                logger.info(f"Downloaded format: {info.get('format')} - {info.get('resolution')} @ {info.get('fps')}fps")
                filename = os.path.join(video_dir, 'video.mp4')
                return filename
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise

    def extract_keyframes(self, video_path, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        frames = []
        features = []
        timestamps = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.size == 0:
                logger.warning("Empty frame detected. Skipping.")
                continue
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            frames.append(frame)
            features.append(self.extract_features(frame))
            timestamps.append(timestamp)
        cap.release()

        if not frames:
            logger.warning("No valid frames extracted from the video.")
            return [], []

        features = np.array(features)
        kmeans = KMeans(n_clusters=num_frames, random_state=42)
        kmeans.fit(features)

        centroids = kmeans.cluster_centers_
        closest_frames = []
        closest_timestamps = []

        for centroid in centroids:
            distances = np.linalg.norm(features - centroid, axis=1)
            closest_frame_idx = np.argmin(distances)
            closest_frames.append(frames[closest_frame_idx])
            closest_timestamps.append(timestamps[closest_frame_idx])

        return self.filter_unique_frames(closest_frames, closest_timestamps)

    def filter_unique_frames(self, frames, timestamps, similarity_threshold=0.85):
        unique_frames = [frames[0]]
        unique_timestamps = [timestamps[0]]

        for i in range(1, len(frames)):
            is_unique = True
            for unique_frame in unique_frames:
                try:
                    if frames[i].shape[0] < 7 or frames[i].shape[1] < 7 or unique_frame.shape[0] < 7 or unique_frame.shape[1] < 7:
                        logger.warning(f"Frame {i} or unique frame is too small. Skipping SSIM comparison.")
                        continue
                    
                    similarity = ssim(frames[i], unique_frame, multichannel=True, channel_axis=2)
                    if similarity > similarity_threshold:
                        is_unique = False
                        break
                except ValueError as e:
                    logger.warning(f"Error comparing frame {i}: {str(e)}. Skipping this comparison.")
                    continue
            if is_unique:
                unique_frames.append(frames[i])
                unique_timestamps.append(timestamps[i])

        return unique_frames, unique_timestamps

    def perform_ocr(self, frame):
        try:
            logger.info("Starting OCR process...")
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocessing
            height, width = frame_rgb.shape[:2]
            if max(height, width) > 2000:
                scale = 2000 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Ensure frame is in numpy array format
            frame_array = np.array(frame_rgb)
            
            # Perform OCR (DocTR expects a list of numpy arrays)
            result = self.ocr_model([frame_array])
            
            # Extract text with confidence
            all_texts = []
            
            # Process each page (in this case, one frame = one page)
            for page in result.pages:
                # Process each block (tables, paragraphs, etc.)
                for block in page.blocks:
                    # Process each line
                    for line in block.lines:
                        line_text = []
                        
                        for word in line.words:
                            line_text.append(word.value)
                        
                        if line_text:
                            text = ' '.join(line_text)
                            all_texts.append(text)
            
            final_text = ' '.join(all_texts).strip()
            logger.info(f"Detected text: {final_text}")
            return final_text
            
        except Exception as e:
            logger.error(f"Error in OCR: {str(e)}", exc_info=True)
            return ""

    def save_keyframes(self, frames, timestamps, video_dir):
        frames_dir = os.path.join(video_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        frame_data = []
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            path = os.path.join(frames_dir, f'frame_{i}.jpg')
            cv2.imwrite(path, frame)
            ocr_text = self.perform_ocr(frame)
            frame_data.append({
                'path': path,
                'timestamp': timestamp,
                'ocr_text': ocr_text
            })
        return frame_data

    def transcribe_audio(self, audio_file):
        result = self.whisper_model.transcribe(audio_file)
        return result["segments"]

    def combine_data(self, transcript, frame_data):
        logger.info(f"Combining data: {len(transcript)} transcript segments and {len(frame_data)} frames")
        combined_data = []
        for segment in transcript:
            combined_data.append({
                'type': 'transcript',
                'start': segment.get('start'),
                'end': segment.get('end'),
                'text': segment.get('text', '')
            })
        
        for frame in frame_data:
            combined_data.append({
                'type': 'frame',
                'timestamp': frame.get('timestamp'),
                'ocr_text': frame.get('ocr_text', ''),
                'path': frame.get('path', '')
            })
        
        combined_data.sort(key=lambda x: x.get('start') or x.get('timestamp') or 0)
        return combined_data

    def prepare_combined_transcript(self, combined_data):
        formatted_data = "Video Content:\n\n"
        for item in combined_data:
            if item['type'] == 'transcript':
                timestamp = timedelta(seconds=item['start'])
                formatted_data += f"[{timestamp}] Transcript: {item['text']}\n"
            else:
                timestamp = timedelta(milliseconds=item['timestamp'])
                formatted_data += f"[{timestamp}] Frame OCR: {item['ocr_text']}\n"
        return formatted_data

    def download_audio_only(self, url, video_dir):
        """Download only audio for normal mode"""
        audio_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(video_dir, 'video'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
            'audioformat': 'mp3',
            'audioquality': '0',
            'format_sort': [
                'acodec:mp3',
                'abr',
                'asr'
            ]
        }
        
        try:
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                logger.info("Downloading audio only...")
                ydl.download([url])
                return os.path.join(video_dir, 'video.mp3')
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise

    def process_video(self, url, mode="normal"):
        try:
            video_dir = self.get_video_output_dir(url)
            logger.info(f"Processing video in {mode} mode in directory: {video_dir}")
            
            if mode == "advanced":
                # For advanced mode, download high quality video
                logger.info("Advanced mode: Downloading high quality video...")
                video_file = self.download_video(url)
                
                # Extract audio from video
                logger.info(f"Extracting audio from {video_file}")
                audio_file = os.path.join(video_dir, 'video.mp3')
                
                audio_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(video_dir, 'video'),
                    'keepvideo': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '320',
                    }],
                    'audioformat': 'mp3',
                    'audioquality': '0',
                    'format_sort': [
                        'acodec:mp3',
                        'abr',
                        'asr'
                    ]
                }
                
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    ydl.download([url])
            else:
                # For normal mode, only download audio
                logger.info("Normal mode: Downloading audio only...")
                audio_file = self.download_audio_only(url, video_dir)
                video_file = None  # No video file in normal mode

            logger.info("Transcribing audio")
            transcript = self.transcribe_audio(audio_file)
            
            if mode == "advanced" and video_file:
                # Only process frames in advanced mode
                logger.info("Advanced mode: Processing frames")
                frames, timestamps = self.extract_keyframes(video_file)
                frame_data = self.save_keyframes(frames, timestamps, video_dir) if frames else []
                combined_data = self.combine_data(transcript, frame_data)
            else:
                # For normal mode, only use transcript
                logger.info("Normal mode: Using transcript only")
                combined_data = self.combine_data(transcript, [])
            
            combined_transcript = self.prepare_combined_transcript(combined_data)
            
            transcript_file = os.path.join(video_dir, 'video_transcript.txt')
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(combined_transcript)

            # Generate and save title
            title = self.gemini_chatbot.generate_title(combined_transcript)
            title_path = os.path.join(video_dir, 'title.txt')
            with open(title_path, 'w', encoding='utf-8') as f:
                f.write(title)

            return {
                "video_file": video_file,
                "audio_file": audio_file,
                "transcript_file": transcript_file,
                "title": title,
                "mode": mode
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
            
    def get_video_id(self, url):
        """Extract video ID from YouTube URL"""
        try:
            if 'youtu.be' in url:
                return url.split('/')[-1]
            elif 'watch?v=' in url:
                return url.split('watch?v=')[-1].split('&')[0]
            elif 'shorts' in url:
                return url.split('/')[-1]
            else:
                return url.split('/')[-1]
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            raise

video_service = VideoService()