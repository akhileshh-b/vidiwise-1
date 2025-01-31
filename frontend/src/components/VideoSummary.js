import React, { useState, useEffect } from 'react';

export default function VideoSummary({ videoId }) {
    const [summary, setSummary] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (videoId) {
            generateSummary();
        }
    }, [videoId]);

    const generateSummary = async () => {
        try {
            setLoading(true);
            setError(null);
            
            const response = await fetch('http://localhost:8080/start-chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: "Please provide a concise summary of this video content.",
                    videoId: videoId
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate summary');
            }

            const data = await response.json();
            setSummary(data.message);
        } catch (error) {
            console.error('Error generating summary:', error);
            setError('Failed to generate summary. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    if (error) {
        return (
            <div className="text-red-400 p-4 rounded-lg bg-red-900/20">
                {error}
                <button
                    onClick={generateSummary}
                    className="mt-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg text-sm transition-colors"
                >
                    Try Again
                </button>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {loading ? (
                <div className="flex items-center justify-center py-6">
                    <div className="w-6 h-6 border-3 border-accent-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
            ) : (
                <div className="text-gray-300 leading-relaxed text-base">
                    {summary}
                </div>
            )}
        </div>
    );
} 