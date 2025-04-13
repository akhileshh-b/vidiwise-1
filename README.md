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

### Backend Deployment (Heroku)

1. **Create a Heroku account** at [heroku.com](https://heroku.com)

2. **Install Heroku CLI** from [devcenter.heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

3. **Deploy backend**
   ```bash
   cd backend
   heroku login
   heroku create vidiwise-backend
   git subtree push --prefix backend heroku main
   ```

4. **Set environment variables**
   ```bash
   heroku config:set GEMINI_API_KEY=your_api_key
   heroku config:set FRONTEND_URL=https://your-frontend-url.com
   ```

### Frontend Deployment (Vercel)

1. **Create a Vercel account** at [vercel.com](https://vercel.com)

2. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

3. **Deploy frontend**
   ```bash
   cd frontend
   vercel login
   vercel
   ```

4. **Configure environment variables** in Vercel dashboard:
   - `REACT_APP_API_URL`: Your backend API URL

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

