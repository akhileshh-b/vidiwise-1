# Vidiwise

An AI-powered video analysis tool that generates insights and enables interactive conversations about video content.

## Local Development

### Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file in the backend directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your Gemini API key from: https://aistudio.google.com/

### Start Backend
```bash
cd backend
python run.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## Deployment

### Backend (Render.com)
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure environment variables:
   - GEMINI_API_KEY
   - PYTHON_VERSION=3.9
   - PORT=8080

### Frontend (Vercel)
1. Create a new project on Vercel
2. Import your GitHub repository
3. Configure environment variable:
   - REACT_APP_API_URL=your_backend_url

## Features
- Video analysis and insights generation
- Interactive AI chat about video content
- Title management
- Video history tracking

## Live Demo
Backend: https://your-backend.onrender.com
Frontend: https://your-frontend.vercel.app
