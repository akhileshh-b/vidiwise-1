import React, { useState, useEffect } from 'react';
import VideoInput from './components/VideoInput';
import ChatInterface from './components/ChatInterface';
import EmbeddedVideo from './components/EmbeddedVideo';
import VideoSummary from './components/VideoSummary';
import VideoHistory from './components/VideoHistory';
import VideoTitle from './components/VideoTitle';

function App() {
  const [videoId, setVideoId] = useState(null);
  const [showContent, setShowContent] = useState(false);
  const [mode, setMode] = useState('normal');
  const [view, setView] = useState('new'); // 'new' or 'history'
  const [currentTitle, setCurrentTitle] = useState('');

  const fetchVideoTitle = async (videoId) => {
    try {
      const response = await fetch(`http://localhost:8080/video-title/${videoId}`);
      if (response.ok) {
        const data = await response.json();
        setCurrentTitle(data.title);
      }
    } catch (error) {
      console.error('Error fetching video title:', error);
    }
  };

  // Add mode handler
  const handleModeChange = (newMode) => {
    setMode(newMode);
  };

  const renderNavigation = () => (
    <div className="flex items-center justify-center space-x-4 mb-8">
      <button
        onClick={() => setView('new')}
        className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
          view === 'new'
            ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
            : 'bg-dark-700 text-gray-300 hover:bg-dark-600'
        }`}
      >
        New Analysis
      </button>
      <button
        onClick={() => setView('history')}
        className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
          view === 'history'
            ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
            : 'bg-dark-700 text-gray-300 hover:bg-dark-600'
        }`}
      >
        History
      </button>
    </div>
  );

  const handleStartContent = async (show, title) => {
    setShowContent(show);
    if (title) {
      setCurrentTitle(title);
    }
  };

  const handleHistorySelect = async (videoId, videoMode, title) => {
    setVideoId(videoId);
    setMode(videoMode);
    setCurrentTitle(title);
    setShowContent(true);
  };

  useEffect(() => {
    if (videoId && showContent) {
      fetchVideoTitle(videoId);
    }
  }, [videoId, showContent]);

  const renderContent = () => {
    if (mode === 'advanced') {
      return (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Side - Stacked Video and Summary */}
          <div className="lg:col-span-5 space-y-6">
            {/* Video Player */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300"></div>
              <div className="relative bg-dark-800/80 backdrop-blur-xl rounded-xl overflow-hidden">
                {videoId && <EmbeddedVideo videoId={videoId} />}
              </div>
            </div>
            
            {/* Summary */}
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300"></div>
              <div className="relative bg-dark-800/80 backdrop-blur-xl rounded-xl p-6">
                <h3 className="text-xl font-medium text-white mb-4">
                  <span className="bg-gradient-to-r from-purple-500 to-pink-500 text-transparent bg-clip-text">
                    Video Summary
                  </span>
                </h3>
                <VideoSummary videoId={videoId} />
              </div>
            </div>
          </div>

          {/* Right Side - Expanded Chat Interface */}
          <div className="lg:col-span-7 relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-500 to-blue-500 rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300"></div>
            <div className="relative h-full">
              <ChatInterface videoId={videoId} mode={mode} />
            </div>
          </div>
        </div>
      );
    } else {
      return (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column - Compact Video and Summary */}
          <div className="lg:col-span-8 space-y-6">
            {/* Video Player with better positioning */}
            <div className="relative group w-full max-w-3xl mx-auto">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 
                rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300"></div>
              <div className="relative bg-dark-800/80 backdrop-blur-xl rounded-xl overflow-hidden">
                <div className="aspect-w-16 aspect-h-9">
                  {videoId && <EmbeddedVideo videoId={videoId} />}
                </div>
              </div>
            </div>
            
            {/* Summary Card with better spacing */}
            <div className="relative group mt-8">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 
                rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300"></div>
              <div className="relative bg-dark-800/80 backdrop-blur-xl rounded-xl p-8">
                <h3 className="text-2xl font-medium text-white mb-6">
                  <span className="bg-gradient-to-r from-purple-500 to-pink-500 text-transparent bg-clip-text">
                    Video Summary
                  </span>
                </h3>
                <VideoSummary videoId={videoId} />
              </div>
            </div>
          </div>

          {/* Right Column - Chat Interface */}
          <div className="lg:col-span-4 relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-500 to-blue-500 
              rounded-2xl blur opacity-30 group-hover:opacity-40 transition duration-300"></div>
            <div className="relative h-full sticky top-8">
              <ChatInterface videoId={videoId} />
            </div>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] 
      from-dark-800 via-dark-900 to-black overflow-hidden">
      {/* Background animations */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -inset-[10px] opacity-50">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full 
            mix-blend-overlay blur-3xl animate-blob"></div>
          <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full 
            mix-blend-overlay blur-3xl animate-blob animation-delay-2000"></div>
          <div className="absolute bottom-1/4 left-1/3 w-96 h-96 bg-pink-500/10 rounded-full 
            mix-blend-overlay blur-3xl animate-blob animation-delay-4000"></div>
        </div>
      </div>

      <div className="min-h-screen backdrop-blur-3xl relative z-10">
        {/* Header */}
        <header className="fixed top-0 left-0 right-0 z-50">
          <div className="h-[1px] w-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"></div>
          <div className="bg-dark-900/90 backdrop-blur-xl">
            <div className="max-w-7xl mx-auto px-6">
              <div className="flex items-center justify-between h-16">
                {/* Logo */}
                <div className="flex items-center">
                  <div className="relative group">
                    <div className="absolute -inset-2 bg-gradient-to-r from-blue-500/20 to-purple-500/20 
                      rounded-lg blur-sm transition-all duration-300 group-hover:blur-md"></div>
                    <span className="relative text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 
                      text-transparent bg-clip-text hover:from-blue-300 hover:to-purple-300 transition-all duration-300">
                      Vidiwise
                    </span>
                  </div>
                </div>

                {/* Navigation */}
                <div className="flex items-center space-x-6">
                  <nav className="hidden md:flex items-center space-x-6 mr-6">
                    <a href="#" className="text-gray-300 hover:text-white transition-all duration-300 
                      hover:scale-105 transform relative group">
                      About
                      <div className="absolute -bottom-1 left-0 w-0 h-[2px] bg-gradient-to-r from-blue-500 to-purple-500 
                        transition-all duration-300 group-hover:w-full"></div>
                    </a>
                    <a href="#" className="text-gray-300 hover:text-white transition-all duration-300 
                      hover:scale-105 transform relative group">
                      Documentation
                      <div className="absolute -bottom-1 left-0 w-0 h-[2px] bg-gradient-to-r from-blue-500 to-purple-500 
                        transition-all duration-300 group-hover:w-full"></div>
                    </a>
                  </nav>
                  
                  {/* Auth buttons */}
                  <div className="flex items-center space-x-4">
                    <button className="px-4 py-2 text-gray-300 hover:text-white transition-all duration-300 
                      hover:scale-105 transform">
                      Sign In
                    </button>
                    <button className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 
                        rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
                      <div className="relative px-4 py-2 bg-dark-800 rounded-lg leading-none flex items-center">
                        <span className="text-gray-100 group-hover:text-white transition duration-300">
                          Get Started
                        </span>
                        <svg className="w-4 h-4 ml-2 -mr-1 transform group-hover:translate-x-1 transition-transform duration-300" 
                          fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" 
                            d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" 
                            clipRule="evenodd">
                          </path>
                        </svg>
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="pt-20 pb-16">
          <div className="max-w-7xl mx-auto px-6">
            {!showContent ? (
              <div className="max-w-4xl mx-auto">
                {/* Hero Section */}
                <div className="text-center mb-12">
                  <div className="relative inline-block mb-8">
                    <h1 className="text-6xl font-bold">
                      <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 text-transparent bg-clip-text">
                        Vidiwise
                      </span>
                    </h1>
                    <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 w-24 h-1 
                      bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></div>
                  </div>
                  
                  <p className="text-gray-300 text-lg font-light mt-8 max-w-xl mx-auto leading-relaxed">
                    Transform Videos into
                    <span className="text-transparent bg-clip-text bg-gradient-to-r 
                      from-blue-400 to-purple-400 font-medium mx-2">
                      Intelligent
                    </span>
                    Insights
                  </p>
                </div>
                
                {/* Mode Selection */}
                <div className="flex items-center justify-center space-x-4 mb-8">
                  <button
                    onClick={() => setView('new')}
                    className="relative group"
                  >
                    <div className={`absolute -inset-0.5 rounded-lg blur opacity-75 
                      transition duration-300 group-hover:opacity-100
                      ${view === 'new' 
                        ? 'bg-gradient-to-r from-blue-500 to-purple-500' 
                        : 'bg-dark-600'}`}
                    ></div>
                    <div className="relative px-8 py-3 bg-dark-800 rounded-lg leading-none">
                      <span className="text-gray-100 group-hover:text-white transition duration-300">
                        New Analysis
                      </span>
                    </div>
                  </button>

                  <button
                    onClick={() => setView('history')}
                    className="relative group"
                  >
                    <div className={`absolute -inset-0.5 rounded-lg blur opacity-75 
                      transition duration-300 group-hover:opacity-100
                      ${view === 'history' 
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500' 
                        : 'bg-dark-600'}`}
                    ></div>
                    <div className="relative px-8 py-3 bg-dark-800 rounded-lg leading-none">
                      <span className="text-gray-100 group-hover:text-white transition duration-300">
                        History
                      </span>
                    </div>
                  </button>
                </div>
                
                {/* Main Input/History Section */}
                <div className="max-w-4xl mx-auto">
                  {view === 'new' ? (
                    <div className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 
                        rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
                      <div className="relative bg-dark-800 rounded-lg p-8">
                        <h2 className="text-2xl font-medium text-transparent bg-clip-text 
                          bg-gradient-to-r from-purple-400 to-pink-400 mb-8">
                          New Analysis
                        </h2>
                        <VideoInput 
                          onVideoProcess={setVideoId} 
                          onStartContent={handleStartContent}
                          onModeChange={handleModeChange}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 
                        rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
                      <div className="relative bg-dark-800 rounded-lg p-8">
                        <h2 className="text-2xl font-medium text-transparent bg-clip-text 
                          bg-gradient-to-r from-purple-400 to-pink-400 mb-8">
                          Recent Analyses
                        </h2>
                        <VideoHistory onSelectVideo={handleHistorySelect} />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                  <div className="flex items-center space-x-6">
                    <VideoTitle 
                      title={currentTitle} 
                      videoId={videoId}
                      onTitleUpdate={(newTitle) => {
                          setCurrentTitle(newTitle);
                          console.log('Title updated to:', newTitle);
                      }}
                    />
                    <button
                      onClick={() => {
                        setShowContent(false);
                        setView('new');
                      }}
                      className="px-4 py-2 bg-dark-700 hover:bg-dark-600 text-gray-300 rounded-lg 
                        transition-all duration-300 text-sm flex items-center space-x-2"
                    >
                      <span>New Analysis</span>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className={`px-3 py-1 rounded-full text-sm ${
                      mode === 'advanced' 
                        ? 'bg-purple-500/20 text-purple-300' 
                        : 'bg-blue-500/20 text-blue-300'
                    }`}>
                      {mode === 'advanced' ? 'Advanced Mode' : 'Normal Mode'}
                    </span>
                    <span className="text-sm text-gray-400">
                      ID: {videoId}
                    </span>
                  </div>
                </div>

                {/* Dynamic Content based on Mode */}
                {renderContent()}
              </div>
            )}
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-dark-700 bg-dark-900/80 backdrop-blur-md">
          <div className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">© 2024 Vidiwise. All rights reserved.</span>
              <div className="flex space-x-6">
                <a href="#" className="text-gray-400 hover:text-white transition-colors">Privacy</a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">Terms</a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">Contact</a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;