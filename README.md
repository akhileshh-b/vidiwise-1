# Vidiwise - Video Analysis Tool

A modern web application for analyzing YouTube videos using AI to provide intelligent insights and interactive chat capabilities.

## Features

- YouTube video analysis
- AI-powered chat interface
- Advanced mode with visual analysis
- Video history tracking
- Customizable titles

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm 8+
- Google Gemini API key

## Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vidiwise.git
   cd vidiwise
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv

   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Unix/macOS:
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Create .env file from example
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY

   # Start backend server
   python run.py
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. Open `http://localhost:3000` in your browser

## Deployment

### Backend Deployment (Render)

1. **Create a Render account** at [render.com](https://render.com)

2. **Create a new Web Service**
   - Connect your GitHub repository
   - Select the repository and branch to deploy
   - Configure the service:
     - Name: `vidiwise-backend` (or your preferred name)
     - Environment: `Python`
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn app.main:app --host=0.0.0.0 --port=$PORT`
     - Plan: Free (or paid if you need more resources)

3. **Set environment variables** in Render dashboard:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `FRONTEND_URL`: Your frontend URL (once deployed)
   - `STORAGE_DIR`: `video_findings`

4. **Deploy** by clicking "Create Web Service"

### Frontend Deployment (Vercel)

1. **Create a Vercel account** at [vercel.com](https://vercel.com)

2. **Import your GitHub repository**
   - After logging in, click "Add New" → "Project"
   - Select your GitHub repository from the list
   - If you don't see your repository, click "Import Git Repository" and connect your GitHub account

3. **Configure the project**
   - Framework Preset: Select "Create React App" (Vercel should auto-detect this)
   - Root Directory: `frontend` (since your frontend code is in this subdirectory)
   - Build Command: Leave as default (`npm run build`)
   - Output Directory: Leave as default (`build`)
   - Install Command: Leave as default (`npm install`)

4. **Set environment variables**
   - Click "Environment Variables"
   - Add the following variable:
     - Name: `REACT_APP_API_URL`
     - Value: Your Render backend URL (e.g., `https://vidiwise-backend.onrender.com`)
   - Make sure to select all environments (Production, Preview, Development)

5. **Deploy** by clicking "Deploy"

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── core/          # Core configuration
│   │   ├── services/      # Business logic
│   │   └── main.py        # FastAPI application
│   ├── video_findings/    # Video storage
│   └── requirements.txt
│
└── frontend/
    ├── public/
    ├── src/
    │   ├── components/    # React components
    │   └── index.js
    └── package.json
```

## Environment Variables

### Backend (.env)
```env
GEMINI_API_KEY=your_api_key_here
API_HOST=0.0.0.0
API_PORT=8080
FRONTEND_URL=http://localhost:3000
STORAGE_DIR=video_findings
```

### Frontend (.env)
```env
REACT_APP_API_URL=http://localhost:8080
```

## License

MIT

