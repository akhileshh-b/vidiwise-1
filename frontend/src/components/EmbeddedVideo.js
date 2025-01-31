import React from 'react';

export default function EmbeddedVideo({ videoId }) {
    // Extract video ID from URL if needed
    const getEmbedId = (videoId) => {
        if (!videoId) return '';
        if (videoId.includes('youtube.com')) {
            return videoId.split('v=')[1];
        }
        if (videoId.includes('youtu.be')) {
            return videoId.split('/').pop();
        }
        return videoId;
    };

    const embedId = getEmbedId(videoId);

    return (
        <div className="aspect-w-16 aspect-h-9 rounded-lg overflow-hidden">
            <iframe
                src={`https://www.youtube.com/embed/${videoId}`}
                title="YouTube video player"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-full"
            ></iframe>
        </div>
    );
}