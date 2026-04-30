import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { setForm } from './store';
import axios from 'axios';
import './App.css';

function App() {
  const dispatch = useDispatch();
  const form = useSelector(state => state.interaction);
  const [input, setInput] = useState("");

  const handleAISubmit = async () => {
    const res = await axios.post('http://localhost:8000/chat', { message: input });
    dispatch(setForm({ hcpName: "Dr. Smith", topics: input, sentiment: "Positive" }));
  };

  return (
    <div className="app-container" style={{ fontFamily: 'Inter' }}>
      <div className="form-column">
        <h2>Log HCP Interaction</h2>
        <div className="field"><label>HCP Name</label><input value={form.hcpName} readOnly /></div>
        <div className="field"><label>Topics</label><textarea value={form.topics} readOnly /></div>
        <div className="field">
            <label>Sentiment</label>
            <input type="radio" checked={form.sentiment === "Positive"} /> Positive
            <input type="radio" checked={form.sentiment === "Neutral"} /> Neutral
        </div>
      </div>
      <div className="ai-column">
        <h3>AI Assistant</h3>
        <div className="chat-box"><p>Describe the interaction...</p></div>
        <input value={input} onChange={(e) => setInput(e.target.value)} />
        <button onClick={handleAISubmit}>Log</button>
      </div>
    </div>
  );
}
export default App;