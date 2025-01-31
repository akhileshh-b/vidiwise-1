# Vidiwise - AI Video Analysis Tool

An AI-powered video analysis tool that provides intelligent insights and interactive chat capabilities for YouTube videos.

## Features

- 🎥 YouTube video analysis
- 🤖 AI-powered chat interface
- 🔍 Advanced visual analysis mode
- 📚 Video history tracking
- ✏️ Customizable titles

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, TailwindCSS
- **ML/AI**: PyTorch, Whisper, DocTR
- **Database**: File-based storage

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm 8+
- Git

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/vidiwise.git
cd vidiwise
```

2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Unix/macOS
source venv/bin/activate

pip install -r requirements.txt
```

3. Frontend Setup
```bash
cd frontend
npm install
```

4. Environment Setup
```bash
# Copy example env file
cp backend/.env.example backend/.env
# Edit .env with your configurations
```

## Running Locally

1. Start Backend
```bash
cd backend
python run.py
```

2. Start Frontend
```bash
cd frontend
npm start
```

3. Visit `http://localhost:3000`

## Environment Variables

Required environment variables:

- `GEMINI_API_KEY`: Google Gemini API key
- `FRONTEND_URL`: Frontend URL for CORS
- `API_PORT`: Backend port (default: 8080)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License - see the [LICENSE](LICENSE) file for details

