import React, { useState } from 'react';

export default function VideoInput({ onVideoProcess, onStartContent, onModeChange }) {
    const [url, setUrl] = useState('');
    const [mode, setMode] = useState('normal');
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsProcessing(true);
        setError('');

        try {
            const response = await fetch('http://localhost:8080/process-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    url,
                    mode
                }),
            });

            if (!response.ok) {
                throw new Error('Video processing failed');
            }

            const data = await response.json();
            onVideoProcess(data.video_id);
            checkStatus(data.video_id);
        } catch (err) {
            setError(err.message);
            setIsProcessing(false);
        }
    };

    const checkStatus = async (videoId) => {
        try {
            const response = await fetch(`http://localhost:8080/video-status/${videoId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                setIsProcessing(false);
                const titleResponse = await fetch(`http://localhost:8080/video-title/${videoId}`);
                if (titleResponse.ok) {
                    const titleData = await titleResponse.json();
                    onStartContent(true, titleData.title);
                }
            } else if (data.status === 'failed') {
                setError('Processing failed');
                setIsProcessing(false);
            } else {
                setTimeout(() => checkStatus(videoId), 5000);
            }
        } catch (err) {
            setError('Status check failed');
            setIsProcessing(false);
        }
    };

    return (
        <div className="relative bg-dark-800/90 backdrop-blur-xl rounded-lg p-6">
            {/* Mode Selection */}
            <div className="mb-6">
                <label className="block text-base font-medium text-white mb-3">
                    Choose Processing Mode
                </label>
                <div className="grid grid-cols-2 gap-4">
                    <button
                        type="button"
                        onClick={() => setMode('normal')}
                        className={`relative group p-4 rounded-xl transition-all duration-300
                            ${mode === 'normal' 
                                ? 'bg-gradient-to-r from-blue-500 to-purple-500' 
                                : 'bg-dark-700 hover:bg-dark-600'}`}
                    >
                        <div className="relative z-10">
                            <div className="text-lg font-medium text-white mb-2">Normal Mode</div>
                            <p className="text-sm text-gray-300">
                                Quick audio transcription for basic analysis
                            </p>
                        </div>
                    </button>
                    <button
                        type="button"
                        onClick={() => setMode('advanced')}
                        className={`relative group p-4 rounded-xl transition-all duration-300
                            ${mode === 'advanced' 
                                ? 'bg-gradient-to-r from-purple-500 to-pink-500' 
                                : 'bg-dark-700 hover:bg-dark-600'}`}
                    >
                        <div className="relative z-10">
                            <div className="text-lg font-medium text-white mb-2">Advanced Mode</div>
                            <p className="text-sm text-gray-300">
                                Deep analysis with visual and audio processing
                            </p>
                        </div>
                    </button>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-lg font-medium text-white mb-4">
                        Enter YouTube URL
                    </label>
                    <div className="relative">
                        <input
                            type="text"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="Paste YouTube URL here"
                            className="w-full px-6 py-4 bg-dark-700 border-2 border-dark-600 rounded-xl
                                text-white placeholder-gray-500 focus:outline-none focus:border-purple-500
                                transition-all duration-300"
                            disabled={isProcessing}
                        />
                    </div>
                </div>

                {error && (
                    <div className="text-red-400 text-sm bg-red-900/20 px-4 py-2 rounded-lg">
                        {error}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={isProcessing || !url}
                    className="w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-500 
                        text-white rounded-xl font-medium text-lg
                        hover:from-blue-600 hover:to-purple-600 
                        transition-all duration-300 transform hover:scale-[1.02]
                        disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                    {isProcessing ? (
                        <div className="flex items-center justify-center">
                            <div className="w-6 h-6 border-3 border-white border-t-transparent 
                                rounded-full animate-spin mr-3">
                            </div>
                            Processing Video...
                        </div>
                    ) : (
                        'Start Analysis'
                    )}
                </button>
            </form>
        </div>
    );
}
