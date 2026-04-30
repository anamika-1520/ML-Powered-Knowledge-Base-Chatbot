# 🚗 Car Price AI Chatbot-ML-Powered-Knowledge-Base-Chatbot


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your-username/your-app-name)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

An intelligent chatbot for Indian car pricing, resale value estimation, and car knowledge using ML + RAG (Retrieval-Augmented Generation). Predict used-car prices with machine learning or explore car-domain insights with AI-powered knowledge retrieval.
# HERE IS THE LINK OF THE STREAMLIT APPLICATION 
(https://ml-powered-knowledge-base-chatbot-hwcrv6c7vqo74pzuyuncds.streamlit.app/)
## ✨ Features

- **ML Price Prediction**: Estimate used-car prices based on brand, model, year, kilometers, and fuel type.
- **RAG Knowledge Base**: Answer questions about car resale, depreciation, fuel efficiency, and buying tips.
- **Intelligent Routing**: Automatically routes queries to ML or RAG based on intent.
- **Streamlit UI**: Clean, responsive web interface for easy interaction.
- **Production-Ready**: Deployable on Streamlit Cloud, Heroku, or any cloud platform.

## 🏗️ Architecture

- **ML Component**: Scikit-learn Linear Regression model for price prediction.
- **RAG Component**: FAISS vector store with HuggingFace embeddings for knowledge retrieval.
- **Orchestration**: LangGraph for stateful conversation routing.
- **UI**: Streamlit for web app.
- **LLM**: Groq API (Llama 3.3) for natural language responses.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anamika-1520/ML-Powered-Knowledge-Base-Chatbot.git
   cd ML-Powered-Knowledge-Base-Chatbot
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env` for local development.
   - Add your API keys:
     ```bash
     GROQ_API_KEY=your_groq_api_key_here
     HUGGINGFACEHUB_ACCESS_TOKEN=your_huggingface_token_here
     LANGCHAIN_API_KEY=your_langchain_api_key_here
     ```
   - For Streamlit Cloud, configure these values in the app's Secrets section instead of using `.env`.

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

Visit `http://localhost:8501` to interact with the chatbot!

## 📁 Project Structure

```
├── app.py                    # Streamlit web application
├── graph.py                  # LangGraph orchestration logic
├── ml_tool.py                # ML prediction utilities
├── rag_faiss.py              # RAG knowledge base setup
├── car_price_predictor.ipynb # Jupyter notebook for experimentation
├── LinearRegressionModel.pkl # Trained ML model (binary)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for LLM responses (get from [Groq Console](https://console.groq.com))

### Model Details
- **ML Model**: Linear Regression trained on Indian car dataset
- **Supported Brands**: Toyota, Honda, Ford, BMW, Hyundai, Maruti
- **Embeddings**: MiniLM-L6-v2 (HuggingFace)
- **Vector DB**: FAISS for efficient similarity search

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Fork this repo to your GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub and select this repo.
4. Set main file as `app.py`.
5. Add secrets in app settings:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
6. Deploy!

### Other Platforms
- **Heroku**: Add `Procfile` with `web: streamlit run app.py --server.port $PORT`
- **Vercel**: Use their Python runtime.
- **Docker**: Build from included `requirements.txt`.

## 💡 Usage Examples

### Price Prediction
```
User: Predict price of 2021 Honda City Petrol 35,000 km
Bot: Estimated Used-Car Price: Rs. 628,784
     Expected Range: Rs. 597,000 - Rs. 660,000
```

### Knowledge Query
```
User: What factors affect car resale value?
Bot: Car resale depends on brand, year, kilometers, fuel type, service history...
```

### Casual Chat
```
User: How are you?
Bot: I'm here and ready to help with car questions!
```

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/), [Streamlit](https://streamlit.io/), and [Groq](https://groq.com/)
- Car dataset and knowledge base inspired by Indian automotive market research
- Icons from [Streamlit](https://streamlit.io/)

## 📞 Support

For questions or issues:
- Open an [issue](https://github.com/anamika-1520/ML-Powered-Knowledge-Base-Chatbot/issues) on GitHub
- Check the [discussions](https://github.com/anamika-1520/ML-Powered-Knowledge-Base-Chatbot/discussions) tab

---

**Made with ❤️ for car enthusiasts in India**
