# ICU Guardian - Production Deployment Guide

## 🚀 Quick Start

### Backend Setup (FastAPI)

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Run FastAPI server
python main.py
# Server will run on http://localhost:8000
```

### Frontend Setup (React + Vite)

```bash
# Install frontend dependencies
cd frontend
npm install

# Run development server
npm run dev
# App will run on http://localhost:5173
```

### Legacy Streamlit Dashboard

```bash
# Run Streamlit (for comparison/backup)
streamlit run dashboard.py
```

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                         │
│                   (http://localhost:5173)                   │
│  • Real-time vitals display                                 │
│  • Live alert feed                                          │
│  • Vision system controls                                   │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ WebSocket + REST API
                    │
┌───────────────────▼─────────────────────────────────────────┐
│                    FastAPI Backend                          │
│                   (http://localhost:8000)                   │
│  • WebSocket server for real-time updates                  │
│  • REST endpoints for system control                       │
│  • ML model inference                                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴──────────────┐
        │                          │
┌───────▼────────┐       ┌────────▼─────────┐
│ Vision System  │       │   ML Predictor   │
│ (MediaPipe)    │       │   (Scikit-learn) │
│ • Camera feed  │       │   • Vitals gen   │
│ • Alert det    │       │   • Risk pred    │
└────────────────┘       └──────────────────┘
```

---

## 🔧 API Endpoints

### System Status
- `GET /api/health` - Health check
- `GET /api/system/status` - System status and uptime

### Vision Control
- `POST /api/vision/start` - Start vision monitoring
- `POST /api/vision/stop` - Stop vision monitoring
- `GET /api/vision/status` - Get vision status

### Monitoring
- `POST /api/monitoring/start` - Start AI monitoring
- `POST /api/monitoring/stop` - Stop AI monitoring

### Data
- `GET /api/vitals/current` - Get current vitals
- `GET /api/alerts/status` - Get alert status

### WebSocket
- `WS /ws` - Real-time bidirectional communication
  - Receives: `vitals_update`, `alerts_update`, `system_event`
  - Sends: `ping` (keep-alive)

---

## 🔒 Production Deployment

### Environment Variables

Create `.env` files:

**Backend (.env)**
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
ENVIRONMENT=production
LOG_LEVEL=INFO
```

**Frontend (.env)**
```env
VITE_API_URL=https://your-api-domain.com
VITE_WS_URL=wss://your-api-domain.com
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

**docker-compose.yml** (to be created):
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

### Cloud Deployment Options

#### Azure App Service
- Deploy backend as Web App
- Deploy frontend as Static Web App
- Use Azure SignalR for WebSocket scaling

#### AWS
- Backend: Elastic Beanstalk or ECS
- Frontend: S3 + CloudFront
- WebSocket: API Gateway WebSocket API

#### Google Cloud
- Backend: Cloud Run
- Frontend: Firebase Hosting
- WebSocket: Cloud Run with HTTP/2

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm run test
```

### Load Testing
```bash
# Test WebSocket connections
python tests/load_test_websocket.py
```

---

## 📈 Performance Targets

| Metric | Target | Production |
|--------|--------|------------|
| WebSocket latency | < 50ms | TBD |
| API response time | < 100ms | TBD |
| Concurrent users | 100+ | TBD |
| Uptime | 99.9% | TBD |
| Alert detection | < 1s | TBD |

---

## 🔐 Security Checklist

- [ ] Enable HTTPS/WSS in production
- [ ] Implement authentication (JWT tokens)
- [ ] Add rate limiting on API endpoints
- [ ] Sanitize all user inputs
- [ ] Enable CORS only for trusted domains
- [ ] Implement proper error handling (no sensitive data in errors)
- [ ] Add audit logging for all system actions
- [ ] Regular security updates for dependencies

---

## 📚 Tech Stack

### Backend
- **FastAPI**: Modern async Python web framework
- **Uvicorn**: ASGI server with WebSocket support
- **Scikit-learn**: ML model inference
- **MediaPipe**: Vision processing
- **Psutil**: Process management

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Fast build tool
- **TailwindCSS**: Styling
- **Recharts**: Data visualization
- **TanStack Query**: Server state management
- **Lucide Icons**: Icon library

---

## 🐛 Troubleshooting

### WebSocket Connection Failed
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify firewall rules

### Vision System Not Starting
- Check camera permissions
- Verify Python path in backend config
- Check vision process logs

### Frontend Build Errors
- Run `npm install` again
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version (requires 18+)

---

## 📞 Support

- **Issues**: Open GitHub issue
- **Documentation**: See `/docs` folder
- **Contact**: [Your contact info]

---

## 📝 License

[Add your license here]
