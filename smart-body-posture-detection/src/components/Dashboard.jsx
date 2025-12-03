import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import io from "socket.io-client";
import { supabase } from "../../supabase";


function Dashboard() {
  const navigate = useNavigate();
  const [preview, setPreview] = useState(null);
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiAnswer, setAiAnswer] = useState("");
  const [conversation, setConversation] = useState([]); // {role: 'user'|'assistant', text}
  const [chatInput, setChatInput] = useState("");
  const [alerts, setAlerts] = useState([]);
  const [uploadResult, setUploadResult] = useState(null);
  const [summary, setSummary] = useState(null);
  const [streams, setStreams] = useState(["0"]);
  const [currentStream, setCurrentStream] = useState("0");
  const [socket, setSocket] = useState(null);
  const [userId] = useState(localStorage.getItem("userId") || "user001");
  const [recommendations, setRecommendations] = useState([]);

  
  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      if (!data.user) navigate("/login");
    });
  }, []);

  useEffect(() => {
    const s = io("http://localhost:5000");
    setSocket(s);

    s.on("connect", () => {
      fetch("http://localhost:5000/register_user", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_id: 0, user_id: userId, email: `${userId}@example.com`, socket_id: s.id }),
      }).catch(() => {});
    });

    s.on("posture_alert", (data) => {
      setAlerts((prev) => {
        const key = `${data.stream_id}:${data.person_id}`;
        const filtered = prev.filter((a) => `${a.stream_id}:${a.person_id}` !== key);
        return [{ ...data, ts: Date.now() }, ...filtered].slice(0, 10);
      });
    });

    s.on('ai_advice_ready', (data) => {
      try {
        const advice = data.advice || data.ai_advice || '';
        setConversation((prev) => [...prev, { role: 'assistant', text: advice }]);
        setAiAnswer(advice);
      } catch (e) {
        console.warn('ai_advice_ready handler error', e);
      }
    });

    return () => s.disconnect();
  }, [userId]);

  useEffect(() => {
    fetch("http://localhost:5000/streams")
      .then((res) => res.json())
      .then((d) => {
        if (Array.isArray(d.streams) && d.streams.length) {
          setStreams(d.streams);
          setCurrentStream(d.streams[0]);
        }
      })
      .catch(() => {});
  }, []);

  const fetchSummary = async () => {
    try {
      const res = await fetch("http://localhost:5000/analytics");
      setSummary(await res.json());
    } catch {
      setSummary(null);
    }
  };

  const handleSendMessage = async () => {
    const msg = chatInput.trim();
    if (!msg) return;
    // append user message locally
    setConversation((prev) => [...prev, { role: 'user', text: msg }]);
    setChatInput('');
    setAiQuestion(msg);
    setAiAnswer('⏳ Waiting for AI response...');

    try {
      const res = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: msg }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setAiAnswer(`❌ Chat Error: ${data.error || 'No response'}`);
        setConversation((prev) => [...prev, { role: 'assistant', text: `❌ Chat Error: ${data.error || 'No response'}` }]);
        return;
      }
      const answer = data.answer || data.chatbot_response || data.answer_text || 'No answer';
      setAiAnswer(answer);
      setConversation((prev) => [...prev, { role: 'assistant', text: answer }]);
    } catch (e) {
      setAiAnswer('❌ Network error.');
      setConversation((prev) => [...prev, { role: 'assistant', text: '❌ Network error.' }]);
    }
  };

  useEffect(() => {
    fetchSummary();
    const id = setInterval(fetchSummary, 60000);
    return () => clearInterval(id);
  }, []);

  const handlePhotoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setUploadResult("⏳ Analyzing posture with AI...");

    const formData = new FormData();
    formData.append("image", file);
    formData.append("user_id", userId);

    try {
      const res = await fetch("http://localhost:5000/upload_photo", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.error) return setUploadResult(`❌ ${data.error || "Failed to analyze posture"}`);

      if (data.status === "success") {
        let result = `${data.severity === "good" ? "✅ GOOD POSTURE" : "⚠️ NEEDS CORRECTION"}\n\n`;

        result += `📊 Analysis: ${data.label}\n`;
        result += `📈 Confidence: ${(data.confidence * 100).toFixed(1)}%\n`;
        result += `🎯 Severity: ${data.severity.toUpperCase()}\n\n`;

        if (data.ai_recommendation) {
          result += `🤖 AI RECOMMENDATION:\n${data.ai_recommendation}\n\n`;
        } else if (data.suggestions?.length) {
          result += "💡 Quick Tips:\n" + data.suggestions.slice(0, 3).map((s) => `• ${s}`).join("\n");
        }

        setUploadResult(result);
      }
    } catch (err) {
      console.error(err);
      setUploadResult("❌ Network error. Check backend server.");
    }
  };

  const handleChatbotRecommendation = async () => {
    setAiAnswer("⏳ Fetching posture data...");
    setAiQuestion("Get Conversational Recommendation");

    try {
      // First, fetch recommendations from backend
      const recRes = await fetch("http://localhost:5000/recommendation");
      const recData = await recRes.json();
      const recList = recData.recommendations || [];

      if (!recList.length) {
        setAiAnswer("⚠️ No posture data available.");
        setRecommendations([]);
        return;
      }

      // Set recommendations immediately so they show in suggestions box
      console.log("Setting recommendations:", recList);
      setRecommendations(recList);

      // Then get chatbot advice
      const postureSummary = recList
        .map((r) => `Person ${r.person_id}: ${r.label} (${r.severity}). Tips: ${r.tips}`)
        .join(".\n");

      const chatRes = await fetch("http://localhost:5000/get_chatbot_advice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, posture_summary: postureSummary }),
      });

      const chatData = await chatRes.json();

      if (!chatRes.ok || chatData.error) {
        setAiAnswer(`❌ Chatbot Error: ${chatData.error || "Failed to receive AI response."}`);
        return;
      }

      setAiAnswer(chatData.chatbot_response || "No response received.");
    } catch (err) {
      console.error("Error in handleChatbotRecommendation:", err);
      setAiAnswer("❌ Network error.");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    navigate("/login");
  };

  return (
    <div className="flex font-poppins text-white bg-gradient-to-br from-[#1e3c72] to-[#2a5298] min-h-screen">
      <aside className="fixed top-0 left-0 h-full w-64 bg-black bg-opacity-80 flex flex-col p-6 space-y-6">
        <div className="text-2xl font-bold flex items-center gap-2">🏃‍♂️ <span>Posture<span className="text-yellow-400">Care</span></span></div>

        <nav className="flex flex-col space-y-3">
          <button onClick={() => navigate("/")} className="hover:text-yellow-400 text-left">Home</button>
          <button className="text-yellow-400 text-left">Dashboard</button>
          <button onClick={handleLogout} className="mt-6 bg-red-500 hover:bg-red-400 transition text-white py-2 rounded">Logout</button>
        </nav>

        <div className="mt-2">
          <div className="font-semibold mb-1">🎥 Stream</div>
          <select value={currentStream} onChange={(e) => setCurrentStream(e.target.value)} className="w-full text-black p-2 rounded">
            {streams.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div className="mt-auto text-sm text-gray-300">
          <div className="font-semibold mb-1">📊 Today</div>
          {summary ? (
            <>
              <div>Confidence avg: {summary.avg_confidence ?? "-"}</div>
              <div>Bad ratio: {summary.bad_ratio ?? "-"}</div>
            </>
          ) : (
            <div>Loading...</div>
          )}
        </div>
      </aside>

      <section className="ml-64 p-8 flex-1 grid grid-cols-3 gap-6">
        {/* Live Feed */}
        <div className="col-span-2 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 shadow-lg relative">
          <h2 className="text-2xl font-semibold mb-4">📸 Live Camera</h2>
          <img src={`http://localhost:5000/video_feed/${encodeURIComponent(currentStream)}`} className="w-full rounded-xl mb-4 bg-black" />
        </div>

        {/* AI Chatbot Above Suggestions */}
        <div className="col-span-1 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">🤖 AI Chatbot Advice</h2>
          {aiAnswer ? (
            <div className="mt-4 bg-black/50 rounded-lg p-4 max-h-80 overflow-y-auto text-sm whitespace-pre-wrap font-mono">
              <p className="font-bold text-yellow-400 mb-2">Query:</p>
              <p className="mb-4">{aiQuestion}</p>
              <p className="font-bold text-yellow-400 mb-2">Response:</p>
              {aiAnswer}
            </div>
          ) : (
            <div className="text-gray-300">Click below to get detailed AI analysis.</div>
          )}

          <button className="mt-4 w-full px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg transition" onClick={handleChatbotRecommendation}>🎯 Get Recommendations</button>
        </div>

        {/* Suggestions Box */}
        <div className="col-span-1 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">✅ Suggestions</h2>

          {recommendations && recommendations.length > 0 ? (
            <div className="space-y-3">
              {recommendations.map((rec, idx) => (
                <div key={`rec-${rec.person_id}-${idx}`} className="text-xs bg-black/50 p-3 rounded border-l-2 border-blue-500">
                  <div className="font-bold text-blue-300">👤 Person {rec.person_id}</div>
                  <div className={`text-sm font-semibold my-1 ${rec.severity === "severe" ? "text-red-400" : rec.severity === "moderate" ? "text-yellow-300" : "text-green-400"}`}>
                    {rec.label}
                  </div>
                  <div className="text-xs text-gray-400 mb-1">Severity: {rec.severity} | Confidence: {(rec.confidence * 100).toFixed(1)}%</div>
                  <div className="text-gray-300 text-xs whitespace-pre-wrap">{rec.tips}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-400 text-sm">
              <p>No suggestions yet.</p>
              <p className="text-xs mt-2 text-gray-500">Click "Get Recommendations" button to fetch posture data</p>
            </div>
          )}
        </div>


        {/* Photo Upload Under Live Feed */}
        <div className="col-span-2 backdrop-blur-lg bg-white/10 border border-white/20 rounded-xl p-6 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">🖼 Upload Posture Image</h2>
          <input type="file" accept="image/*" onChange={handlePhotoUpload} className="mb-4 bg-white text-black rounded p-2 w-full" />

          {preview && <img src={preview} className="mt-4 max-h-64 rounded-lg mx-auto" />}

          {uploadResult && (
            <div className="mt-4 bg-black/50 border border-yellow-400 rounded-lg p-4 max-h-96 overflow-y-auto text-sm whitespace-pre-wrap font-mono">
              {uploadResult}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default Dashboard;
