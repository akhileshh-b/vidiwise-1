from app.main import app
import uvicorn
import os

if __name__ == "__main__":
    # Update port to use environment variable for deployment
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 