import React, { useState, useEffect } from 'react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

export default function VideoTitle({ title, videoId, onTitleUpdate }) {
    const [isEditing, setIsEditing] = useState(false);
    const [newTitle, setNewTitle] = useState(title || 'Untitled Video');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        setNewTitle(title || 'Untitled Video');
    }, [title]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!videoId) {
            setError('No video ID available');
            return;
        }
        
        setLoading(true);
        setError('');

        try {
            console.log('Updating title for video:', videoId); // Debug log
            const response = await fetch(`${API_URL}/update-title`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    videoId,
                    newTitle: newTitle.trim()
                })
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to update title');
            }

            onTitleUpdate?.(data.title);
            setIsEditing(false);
        } catch (error) {
            console.error('Error updating title:', error);
            setError(error.message);
        } finally {
            setLoading(false);
        }
    };

    if (isEditing) {
        return (
            <div className="space-y-2">
                <form onSubmit={handleSubmit} className="flex items-center gap-2">
                    <input
                        type="text"
                        value={newTitle}
                        onChange={(e) => {
                            setNewTitle(e.target.value);
                            setError('');
                        }}
                        className="flex-1 px-3 py-1 bg-dark-700 border border-dark-600 rounded-lg
                            text-white focus:outline-none focus:border-accent-primary"
                        placeholder="Enter new title"
                        maxLength={50}
                    />
                    <button
                        type="submit"
                        disabled={loading || !newTitle.trim()}
                        className="px-3 py-1 bg-accent-primary text-white rounded-lg 
                            hover:bg-accent-primary/90 disabled:opacity-50"
                    >
                        {loading ? 'Saving...' : 'Save'}
                    </button>
                    <button
                        type="button"
                        onClick={() => {
                            setNewTitle(title || 'Untitled Video');
                            setIsEditing(false);
                            setError('');
                        }}
                        className="px-3 py-1 bg-dark-700 text-gray-300 rounded-lg hover:bg-dark-600"
                    >
                        Cancel
                    </button>
                </form>
                {error && (
                    <div className="text-red-400 text-sm">
                        {error}
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-white">
                {title || 'Untitled Video'}
            </h1>
            <button
                onClick={() => setIsEditing(true)}
                className="p-1 text-gray-400 hover:text-white transition-colors"
                title="Edit title"
            >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                </svg>
            </button>
        </div>
    );
} 