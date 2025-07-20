# KPBU Chatbot Setup Guide

## 🤖 AI-Powered Investment Advisor

API KPBU sekarang dilengkapi dengan chatbot AI yang menggunakan Google Gemini untuk memberikan konsultasi investasi yang personal dan persuasif.

## Setup Requirements

### 1. Gemini API Key

1. **Dapatkan API Key:**
   - Kunjungi [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Login dengan akun Google Anda
   - Klik "Create API Key"
   - Copy API key yang dihasilkan

2. **Set Environment Variable:**
   ```bash
   # Metode 1: Export di terminal
   export GEMINI_API_KEY="your-actual-gemini-api-key-here"
   
   # Metode 2: Buat file .env
   echo "GEMINI_API_KEY=your-actual-gemini-api-key-here" > .env
   ```

3. **Install Dependencies:**
   ```bash
   pip install google-generativeai
   ```

### 2. Start API Server

```bash
cd /Users/ridwanfirmansyah/Documents/GitHub/kpbu-api
python api_matchmaker.py
```

## 🎯 Chatbot Features

### Core Capabilities
- **🎯 Investment-Focused**: Selalu mengarahkan user untuk berinvestasi dengan pendekatan halus
- **📊 Data-Driven**: Menggunakan hasil matching engine untuk rekomendasi personal
- **💬 Natural Language**: Memahami pertanyaan dalam bahasa Indonesia
- **🧠 Context Aware**: Mengingat profil investor dan rekomendasi proyek
- **🔄 Session Management**: Menyimpan context percakapan selama 24 jam

### Investment Consultation Flow
1. **Start Session** → User menekan "Start Chat"
2. **Profile Analysis** → Sistem menganalisis profil investor
3. **Project Matching** → Matching engine menghasilkan rekomendasi
4. **AI Introduction** → Chatbot menyapa dan memperkenalkan proyek
5. **Interactive Q&A** → User bertanya, AI menjawab dengan persuasif
6. **Investment Guidance** → AI mengarahkan ke langkah investasi konkret

## 📝 Quick Start Example

### 1. Start Chat Session
```bash
curl -X POST "http://localhost:8000/chat/start" \\
  -H "Authorization: Bearer kpbu-matchmaker-2025" \\
  -H "Content-Type: application/json" \\
  -d '{
    "investor_profile": {
        "toleransi_risiko": "Moderat",
        "preferensi_sektor": [3, 1],
        "horison_investasi": "Jangka Panjang", 
        "ukuran_investasi": "Besar",
        "limit": 3
    },
    "user_name": "Pak Investor"
  }'
```

### 2. Send Message
```bash
curl -X POST "http://localhost:8000/chat/message" \\
  -H "Authorization: Bearer kpbu-matchmaker-2025" \\
  -H "Content-Type: application/json" \\
  -d '{
    "session_id": "session-id-from-step-1",
    "message": "Berapa return yang bisa saya dapatkan?"
  }'
```

## 🎨 Frontend Integration

### JavaScript/React Example
```javascript
class KPBUChatbot {
    constructor(apiBaseUrl, token) {
        this.apiBaseUrl = apiBaseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
        this.sessionId = null;
    }
    
    async startChat(investorProfile, userName) {
        const response = await fetch(`${this.apiBaseUrl}/chat/start`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                investor_profile: investorProfile,
                user_name: userName
            })
        });
        
        const result = await response.json();
        if (result.success) {
            this.sessionId = result.session_id;
            return result;
        }
        throw new Error(result.detail || 'Failed to start chat');
    }
    
    async sendMessage(message) {
        if (!this.sessionId) {
            throw new Error('Chat session not started');
        }
        
        const response = await fetch(`${this.apiBaseUrl}/chat/message`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                session_id: this.sessionId,
                message: message
            })
        });
        
        const result = await response.json();
        if (result.success) {
            return result;
        }
        throw new Error(result.detail || 'Failed to send message');
    }
}

// Usage
const chatbot = new KPBUChatbot('http://localhost:8000', 'kpbu-matchmaker-2025');

// Start investment consultation
const investorProfile = {
    toleransi_risiko: 'Moderat',
    preferensi_sektor: [3, 1, 16],
    horison_investasi: 'Jangka Panjang',
    ukuran_investasi: 'Besar',
    limit: 5
};

chatbot.startChat(investorProfile, 'Pak Budi')
    .then(result => {
        console.log('AI:', result.message);
        console.log('Recommendations:', result.recommendations);
        console.log('Suggested Questions:', result.suggested_questions);
    });

// Continue conversation
chatbot.sendMessage('Berapa minimum investasi yang diperlukan?')
    .then(result => {
        console.log('AI:', result.message);
        console.log('Follow-up Questions:', result.suggested_questions);
    });
```

### React Component Example
```jsx
import React, { useState, useEffect } from 'react';

const KPBUChatInterface = () => {
    const [chatbot, setChatbot] = useState(null);
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        // Initialize chatbot
        const bot = new KPBUChatbot('http://localhost:8000', 'kpbu-matchmaker-2025');
        setChatbot(bot);
    }, []);

    const startChat = async () => {
        if (!chatbot) return;
        
        setIsLoading(true);
        try {
            const investorProfile = {
                toleransi_risiko: 'Moderat',
                preferensi_sektor: [3, 1, 16],
                horison_investasi: 'Jangka Panjang',
                ukuran_investasi: 'Besar',
                limit: 5
            };
            
            const result = await chatbot.startChat(investorProfile, 'Investor');
            
            setMessages([{
                type: 'ai',
                content: result.message,
                recommendations: result.recommendations
            }]);
            setSuggestions(result.suggested_questions || []);
        } catch (error) {
            console.error('Error starting chat:', error);
        }
        setIsLoading(false);
    };

    const sendMessage = async (message) => {
        if (!chatbot || !message.trim()) return;
        
        setMessages(prev => [...prev, { type: 'user', content: message }]);
        setInputMessage('');
        setIsLoading(true);
        
        try {
            const result = await chatbot.sendMessage(message);
            setMessages(prev => [...prev, { type: 'ai', content: result.message }]);
            setSuggestions(result.suggested_questions || []);
        } catch (error) {
            console.error('Error sending message:', error);
        }
        setIsLoading(false);
    };

    return (
        <div className="kpbu-chat-interface">
            <div className="chat-header">
                <h3>🤖 KPBU Investment Advisor</h3>
                {messages.length === 0 && (
                    <button onClick={startChat} disabled={isLoading}>
                        Start Investment Consultation
                    </button>
                )}
            </div>
            
            <div className="chat-messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`}>
                        <div className="message-content">
                            {msg.content}
                        </div>
                        {msg.recommendations && (
                            <div className="recommendations">
                                <h4>Recommended Projects:</h4>
                                {msg.recommendations.map((proj, i) => (
                                    <div key={i} className="project-card">
                                        <strong>{proj.nama_proyek}</strong>
                                        <span>Risk: {proj.profil_risiko}</span>
                                        <span>Investment: Rp {proj.nilai_investasi_triliun}T</span>
                                        <span>Match: {proj.skor_kecocokan_persen}%</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && <div className="message ai loading">AI is typing...</div>}
            </div>
            
            {suggestions.length > 0 && (
                <div className="suggested-questions">
                    <h4>💡 Suggested Questions:</h4>
                    {suggestions.map((question, index) => (
                        <button 
                            key={index}
                            onClick={() => sendMessage(question)}
                            className="suggestion-btn"
                        >
                            {question}
                        </button>
                    ))}
                </div>
            )}
            
            <div className="chat-input">
                <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage(inputMessage)}
                    placeholder="Ask about KPBU investments..."
                    disabled={isLoading || messages.length === 0}
                />
                <button 
                    onClick={() => sendMessage(inputMessage)}
                    disabled={isLoading || !inputMessage.trim() || messages.length === 0}
                >
                    Send
                </button>
            </div>
        </div>
    );
};

export default KPBUChatInterface;
```

## 🔧 Configuration Options

### Customizing AI Behavior
Edit the system prompt in `api_matchmaker.py` untuk mengubah:
- Gaya komunikasi (formal/casual)
- Tingkat persuasiveness
- Focus area (ROI, sustainability, government backing, etc.)
- Language preferences

### Session Management
- **Default Session Duration**: 24 jam
- **Auto Cleanup**: Jalankan `/chat/cleanup` secara periodik
- **Memory**: Sessions disimpan in-memory (production: gunakan Redis/Database)

### Rate Limiting
Untuk production, tambahkan rate limiting:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat/message")
@limiter.limit("10/minute")  # Max 10 messages per minute
async def send_chat_message(request: Request, ...):
    # existing code
```

## 🚀 Production Deployment

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your-production-gemini-key

# Optional
DATABASE_URL=postgresql://user:pass@host:port/dbname
REDIS_URL=redis://host:port/db
API_HOST=0.0.0.0
API_PORT=8000
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api_matchmaker:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Performance Monitoring
- Monitor Gemini API usage dan quotas
- Track chat session metrics
- Log conversation quality untuk improvement

## 🎯 Investment Strategy

Chatbot dirancang dengan strategi persuasif yang halus:

### 1. **Trust Building**
- Menyapa dengan professional
- Menampilkan rekomendasi yang relevan
- Memberikan data dan fakta konkret

### 2. **Value Proposition**
- Menekankan ROI yang menarik (12-15% vs 3-4% deposito)
- Highlight government backing dan stability
- Sustainability dan ESG compliance

### 3. **Objection Handling**
- Jika ditanya risiko → Fokus pada mitigasi dan track record
- Jika ragu minimum investasi → Tawarkan berbagai opsi
- Jika bandingkan dengan saham → Highlight predictability dan government guarantee

### 4. **Call to Action**
- Gentle nudging ke next steps
- Tawarkan detailed information
- Facilitate introduction ke investment team

Chatbot ini dirancang untuk menjadi **first touchpoint** yang effective dalam investor journey, mengkonversi interest menjadi qualified leads untuk tim investment.

## 📞 Support

Jika ada issues dengan chatbot:
1. Check Gemini API key validity
2. Verify internet connection untuk API calls
3. Monitor API rate limits
4. Check session management untuk memory issues

Happy investing! 🚀💰
