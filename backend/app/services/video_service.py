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
import shutil
from pytube import YouTube
import io
from PIL import Image
import sys
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from .gemini_service import GeminiChatbot
from app.core.config import STORAGE_DIR, GEMINI_API_KEY
# Fix the moviepy import
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback import if the first one fails
    try:
        import moviepy.editor as mp
        VideoFileClip = mp.VideoFileClip
    except ImportError:
        # Last resort
        from moviepy.video.io.VideoFileClip import VideoFileClip

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
        
        # First try with pytube for YouTube Shorts
        try:
            logger.info("Attempting to download with pytube first...")
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if stream:
                logger.info(f"Found stream: {stream.resolution} - {stream.mime_type}")
                output_path = stream.download(output_path=video_dir, filename="video.mp4")
                logger.info(f"Downloaded to: {output_path}")
                return output_path
            else:
                logger.warning("No suitable stream found with pytube, falling back to yt-dlp")
        except Exception as e:
            logger.warning(f"Pytube download failed: {str(e)}, falling back to yt-dlp")
        
        # Fallback to yt-dlp with simplified options
        ydl_opts = {
            'format': 'best',  # Just get the best available format
            'outtmpl': os.path.join(video_dir, 'video.%(ext)s'),
            # Remove FFmpeg dependency
            'postprocessors': [],
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'youtube_include_dash_manifest': False,
            'youtube_include_hls_manifest': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],  # Use Android client which is more reliable
                    'player_skip': ['js', 'configs', 'webpage'],  # Skip problematic parts
                }
            }
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Starting video download with yt-dlp...")
                info = ydl.extract_info(url, download=True)
                logger.info(f"Downloaded format: {info.get('format')} - {info.get('resolution')} @ {info.get('fps')}fps")
                
                # Get the downloaded file path
                downloaded_file = ydl.prepare_filename(info)
                
                # If the file is not already MP4, convert it using moviepy
                if not downloaded_file.endswith('.mp4'):
                    logger.info(f"Converting {downloaded_file} to MP4 using moviepy...")
                    output_path = os.path.join(video_dir, 'video.mp4')
                    video = VideoFileClip(downloaded_file)
                    video.write_videofile(output_path, codec='libx264', audio_codec='aac')
                    video.close()
                    
                    # Remove the original file if it's different from the output
                    if downloaded_file != output_path and os.path.exists(downloaded_file):
                        os.remove(downloaded_file)
                    
                    return output_path
                else:
                    # If it's already MP4, just rename it to our standard name
                    output_path = os.path.join(video_dir, 'video.mp4')
                    if downloaded_file != output_path:
                        shutil.move(downloaded_file, output_path)
                    return output_path
        except Exception as e:
            logger.error(f"Error downloading video with yt-dlp: {str(e)}")
            
            # Last resort: try with minimal options
            try:
                logger.info("Trying last resort download with minimal options...")
                minimal_opts = {
                    'format': 'best',
                    'outtmpl': os.path.join(video_dir, 'video.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android'],
                            'player_skip': ['js', 'configs', 'webpage'],
                        }
                    }
                }
                with yt_dlp.YoutubeDL(minimal_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    logger.info(f"Last resort download successful")
                    
                    # Get the downloaded file path
                    downloaded_file = ydl.prepare_filename(info)
                    
                    # If the file is not already MP4, convert it using moviepy
                    if not downloaded_file.endswith('.mp4'):
                        logger.info(f"Converting {downloaded_file} to MP4 using moviepy...")
                        output_path = os.path.join(video_dir, 'video.mp4')
                        video = VideoFileClip(downloaded_file)
                        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
                        video.close()
                        
                        # Remove the original file if it's different from the output
                        if downloaded_file != output_path and os.path.exists(downloaded_file):
                            os.remove(downloaded_file)
                        
                        return output_path
                    else:
                        # If it's already MP4, just rename it to our standard name
                        output_path = os.path.join(video_dir, 'video.mp4')
                        if downloaded_file != output_path:
                            shutil.move(downloaded_file, output_path)
                        return output_path
            except Exception as last_e:
                logger.error(f"All download attempts failed: {str(last_e)}")
                raise

    def extract_keyframes(self, video_path, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        frames = []
        features = []
        timestamps = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps} fps, {duration:.2f} seconds")
        
        # Calculate frame interval
        interval = max(1, total_frames // num_frames)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                frames.append(frame)
                features.append(self.extract_features(frame))
                timestamps.append(frame_idx / fps)
                
            frame_idx += 1
            
        cap.release()
        
        if not frames:
            logger.warning("No frames extracted from video")
            return [], []
            
        # Initialize closest_frames and closest_timestamps
        closest_frames = []
        closest_timestamps = []
            
        # Cluster frames by feature similarity
        if len(frames) > num_frames:
            features_array = np.array(features)
            kmeans = KMeans(n_clusters=num_frames, random_state=42)
            clusters = kmeans.fit_predict(features_array)
            
            # Select frames closest to cluster centers
            for i in range(num_frames):
                cluster_frames = [j for j, c in enumerate(clusters) if c == i]
                if not cluster_frames:
                    continue
                    
                cluster_features = features_array[cluster_frames]
                center = kmeans.cluster_centers_[i]
                
                # Find frame closest to cluster center
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_frame_idx = cluster_frames[np.argmin(distances)]
                
                closest_frames.append(frames[closest_frame_idx])
                closest_timestamps.append(timestamps[closest_frame_idx])
        else:
            # If we have fewer frames than requested, use all frames
            closest_frames = frames
            closest_timestamps = timestamps

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
            
            # Perform OCR using DocTR
            result = self.ocr_model([frame_array])
            
            # Extract text from result
            extracted_text = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            extracted_text += word.value + " "
                        extracted_text += "\n"
                    extracted_text += "\n"
            
            logger.info(f"OCR completed. Extracted {len(extracted_text)} characters.")
            return extracted_text.strip()
        except Exception as e:
            logger.error(f"Error in OCR process: {str(e)}")
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