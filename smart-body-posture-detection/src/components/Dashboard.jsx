import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

function Dashboard() {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [preview, setPreview] = useState(null);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");

  // Redirect if not logged in
  useEffect(() => {
    const isLoggedIn = localStorage.getItem("isLoggedIn");
    if (isLoggedIn !== "true") {
      navigate("/login");
    }
  }, [navigate]);

  // Start camera
  const startCamera = async () => {
    try {
      const newStream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = newStream;
      setStream(newStream);
    } catch (err) {
      alert("Camera access denied or error: " + err.message);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setStream(null);
    }
  };

  // Handle file upload
  const handlePhotoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setPreview(url);
    }
  };

  // AI Assistant (dummy)
  const handleAsk = () => {
    if (!aiQuestion.trim()) return;
    setAiAnswer("🤖 Thinking...");
    setTimeout(() => {
      setAiAnswer(`🧠 AI: Great question! You asked: "${aiQuestion}"`);
    }, 800);
  };

  // Logout
  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    navigate("/login");
  };

  return (
    <div className="flex font-poppins text-white bg-gradient from-[#1e3c72] to-[#2a5298] min-h-screen">
      {/* Sidebar */}
      <aside className="fixed top-0 left-0 h-full w-64 bg-black bg-opacity-80 flex flex-col p-6 space-y-8">
        <div className="text-2xl font-bold flex items-center gap-2">
          🏃‍♂️ <span>Posture<span className="text-yellow-400">Care</span></span>
        </div>
        <nav className="flex flex-col space-y-3">
          <button onClick={() => navigate("/")} className="hover:text-yellow-400 text-left">Home</button>
          <button className="text-yellow-400 text-left">Dashboard</button>
          <button onClick={handleLogout} className="mt-6 bg-red-500 hover:bg-red-400 transition text-white py-2 rounded">
            Logout
          </button>
        </nav>
      </aside>

      {/* Main Dashboard */}
      <section className="ml-64 p-8 flex-1">
        <h1 className="text-4xl font-semibold mb-6">Welcome to Your Dashboard 👋</h1>

        {/* Camera Section */}
        <div className="backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 mb-8 shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">📸 Live Camera (Posture Detection)</h2>

        {/* Flask video feed */}
        <img
            src="http://localhost:5000/video_feed"
            alt="Posture Feed"
            className="w-full rounded-xl mb-4 bg-black"
        />

        <p className="text-sm text-gray-300">
            Keep your posture in frame — this view is processed in real time using Mediapipe.
        </p>
        </div>


        {/* Upload Section */}
        <div className="backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 mb-8 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">🖼 Upload Posture Image</h2>
          <input type="file" onChange={handlePhotoUpload} accept="image/*" className="mb-4 bg-white text-black rounded p-2 w-full" />
          {preview && <img src={preview} alt="Preview" className="mt-4 max-h-64 rounded-lg mx-auto" />}
        </div>

        {/* AI Assistant */}
        <div className="backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">💬 Ask the AI Assistant</h2>
          <div className="flex space-x-2">
            <input
              type="text"
              value={aiQuestion}
              onChange={(e) => setAiQuestion(e.target.value)}
              placeholder="Ask a posture or health question..."
              className="flex-1 px-4 py-2 rounded text-black"
            />
            <button
              onClick={handleAsk}
              className="bg-yellow-500 text-black px-4 py-2 rounded font-semibold hover:bg-yellow-400 transition"
            >
              Ask
            </button>
          </div>
          {aiAnswer && <div className="mt-4 text-lg text-gray-200">{aiAnswer}</div>}
        </div>
        <button
            className="mt-6 px-6 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg"
            onClick={() => {
            fetch("http://localhost:5000/recommendation")
                .then(res => res.json())
                .then(data => alert(data.message));
            }}
        >
            Get Recommendation
      </button>
      </section>
    </div>
  );
}

export default Dashboard;
