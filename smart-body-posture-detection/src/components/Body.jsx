import React from 'react'
import { useNavigate } from 'react-router-dom'

function Body() {
  const navigate = useNavigate()
  return (
    <>
      {/* Hero Section */}
      <section id="home" className="  bg-gray-900 text-white   min-h-screen flex flex-col items-center justify-center text-center px-4 relative">
        <h1 className="text-5xl font-bold mb-4">
          Correct Your Posture,{" "}
          <span className="text-yellow-400">Stay Healthy</span>
        </h1>
        <p className="text-gray-200 max-w-2xl mb-8">
          Real-time posture detection and guidance for students and professionals.
        </p>
        <button onClick={() => navigate('/login')} className="bg-yellow-500 text-black px-6 py-3 rounded-lg font-semibold text-lg hover:bg-yellow-400 transition">Check Your Posture</button>
      </section>

      {/* Featured Features Section */}
      <section id="features" className="  bg-gray-900 text-white  py-16 px-6 text-center">
        <h2 className="text-3xl font-semibold mb-10">🌟 Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 justify-items-center">
          <div className="card rounded-xl p-4 shadow-lg max-w-sm">
            <img
              src="https://cdn-icons-png.flaticon.com/512/4149/4149674.png"
              alt="Posture Detection"
              className="mx-auto h-32 mb-4"
            />
            <h3 className="text-xl font-semibold mb-2">Posture Detection</h3>
            <p className="text-gray-700 mb-4">
              Detect your posture instantly using AI.
            </p>
          </div>

          <div className="card rounded-xl p-4 shadow-lg max-w-sm">
            <img
              src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png"
              alt="Posture History"
              className="mx-auto h-32 mb-4"
            />
            <h3 className="text-xl font-semibold mb-2">Posture History</h3>
            <p className="text-gray-700 mb-4">
              Track your progress over time.
            </p>
          </div>

          <div className="card rounded-xl p-4 shadow-lg max-w-sm">
            <img
              src="https://cdn-icons-png.flaticon.com/512/3209/3209265.png"
              alt="Health Tips"
              className="mx-auto h-32 mb-4"
            />
            <h3 className="text-xl font-semibold mb-2">Health Tips</h3>
            <p className="text-gray-700 mb-4">
              Get personalized correction tips.
            </p>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how" className="py-16 px-6 bg-black bg-opacity-40 text-center">
        <h2 className="text-3xl font-semibold text-gray-300 mb-10">🛠️ How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 justify-items-center">
          <div className="max-w-xs">
            <img
              src="https://cdn-icons-png.flaticon.com/512/1041/1041916.png"
              alt="Capture Posture"
              className="mx-auto h-20 mb-3"
            />
            <h3 className="text-xl font-semibold mb-1">Step 1: Capture Posture</h3>
            <p className="text-gray-300">
              Use your camera to capture your posture.
            </p>
          </div>

          <div className="max-w-xs">
            <img
              src="https://cdn-icons-png.flaticon.com/512/4710/4710746.png"
              alt="Analyze"
              className="mx-auto h-20 mb-3"
            />
            <h3 className="text-xl font-semibold mb-1">Step 2: Analyze</h3>
            <p className="text-gray-300">
              Our AI analyzes body keypoints instantly.
            </p>
          </div>

          <div className="max-w-xs">
            <img
              src="https://cdn-icons-png.flaticon.com/512/456/456212.png"
              alt="Get Feedback"
              className="mx-auto h-20 mb-3"
            />
            <h3 className="text-xl font-semibold mb-1">Step 3: Get Feedback</h3>
            <p className="text-gray-300">
              Receive posture correction tips in real time.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}

export default Body