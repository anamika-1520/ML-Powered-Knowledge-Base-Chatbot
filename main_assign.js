// frontend/src/components/LogInteractionScreen.js
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import './LogInteraction.css'; // Ensure font-family: 'Inter', sans-serif; 

const LogInteractionScreen = () => {
    const dispatch = useDispatch();
    const formData = useSelector((state) => state.interaction);

    return (
        <div className="container">
            <header><h2>Log HCP Interaction</h2></header> {/* [cite: 22] */}
            
            <div className="main-layout">
                {/* Left Side: Structured Form [cite: 12, 23] */}
                <div className="form-section">
                    <label>HCP Name</label>
                    <input type="text" placeholder="Search or select HCP..." /> {/* [cite: 25] */}
                    
                    <label>Topics Discussed</label>
                    <textarea placeholder="Enter key discussion points..."></textarea> {/* [cite: 40] */}
                    
                    <div className="sentiment-radio">
                        <label>Sentiment:</label>
                        <input type="radio" name="sentiment" value="Positive" /> Positive
                        <input type="radio" name="sentiment" value="Neutral" /> Neutral
                        <input type="radio" name="sentiment" value="Negative" /> Negative
                    </div> {/* [cite: 46] */}
                </div>

                {/* Right Side: AI Assistant Chat [cite: 31, 32] */}
                <div className="ai-assistant">
                    <h3>AI Assistant</h3>
                    <div className="chat-box">
                        <p>Log interaction details here (e.g., "Met Dr. Smith...")</p> {/* [cite: 33, 34] */}
                    </div>
                    <input type="text" placeholder="Describe interaction..." /> {/* [cite: 56] */}
                    <button>Log</button> {/* [cite: 57] */}
                </div>
            </div>
        </div>
    );
};

export default LogInteractionScreen;