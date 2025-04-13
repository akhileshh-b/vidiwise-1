import React, { useState, useEffect } from 'react';

export default function VideoHistory({ onSelectVideo }) {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetchHistory = async () => {
        try {
            const response = await fetch('http://localhost:8080/video-history');
            if (response.ok) {
                const data = await response.json();
                setHistory(data);
            }
        } catch (error) {
            console.error('Error fetching history:', error);
        } finally {
            setLoading(false);
        }
    };

    // Refresh history periodically to catch title updates
    useEffect(() => {
        fetchHistory();
        const interval = setInterval(fetchHistory, 5000); // Refresh every 5 seconds
        return () => clearInterval(interval);
    }, []);

    const handleVideoSelect = async (videoId, mode, title) => {
        try {
            const response = await fetch(`http://localhost:8080/load-historical-video/${videoId}`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to load video');
            }

            onSelectVideo(videoId, mode, title);
        } catch (error) {
            console.error('Error loading historical video:', error);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center py-8">
                <div className="w-8 h-8 border-4 border-accent-primary border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {history.map((video) => (
                <div
                    key={video.id}
                    onClick={() => handleVideoSelect(video.id, video.mode, video.title)}
                    className="p-4 bg-dark-700/50 rounded-lg cursor-pointer 
                        hover:bg-dark-600/50 transition-all duration-300"
                >
                    <div className="flex items-center space-x-4">
                        <div className="w-24 h-16 rounded-lg overflow-hidden">
                            <img 
                                src={`https://img.youtube.com/vi/${video.id}/mqdefault.jpg`}
                                alt="Thumbnail"
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className="flex-1">
                            <h3 className="text-white font-medium truncate">{video.title}</h3>
                            <div className="flex items-center space-x-3 mt-1">
                                <span className={`px-2 py-0.5 rounded-full text-xs ${
                                    video.mode === 'advanced' 
                                        ? 'bg-purple-500/20 text-purple-300' 
                                        : 'bg-blue-500/20 text-blue-300'
                                }`}>
                                    {video.mode === 'advanced' ? 'Advanced' : 'Normal'}
                                </span>
                                <span className="text-gray-400 text-sm">
                                    {new Date(video.timestamp).toLocaleDateString()}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
} 