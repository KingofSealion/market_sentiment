# Agri Commodities Sentiment Dashboard

A comprehensive web application for tracking agricultural commodities market sentiment with real-time data visualization and AI-powered analysis.

## 🌟 Features

### Dashboard Tab
- **Sentiment Score Cards**: Real-time sentiment scores for 6 major commodities (Corn, Wheat, Soybean, Soybean Meal, Soybean Oil, Palm Oil)
- **Time Series Chart**: Interactive dual Y-axis chart showing sentiment scores and price trends
- **News & Analysis**: Latest news articles with sentiment analysis and reasoning
- **Trending Keywords**: Most frequently mentioned keywords with occurrence counts

### AI Chatbot Tab
- **Streaming Chat Interface**: ChatGPT-style interface with real-time streaming responses
- **Agricultural Market Expert**: AI analyst specialized in commodity market sentiment
- **Interactive Conversations**: Ask questions about market trends, specific commodities, or sentiment analysis

### Technical Features
- **Responsive Design**: Mobile and desktop optimized
- **Loading States**: Skeleton UI for better UX during data loading
- **Error Handling**: Comprehensive error messages and retry mechanisms
- **Real-time Updates**: Live data from PostgreSQL database
- **Modern UI**: Material-UI components with Tailwind CSS styling

## 🏗️ Architecture

### Frontend
- **Framework**: Next.js 15 with React 19 and TypeScript
- **UI Library**: Material-UI (MUI) v7
- **Charts**: Recharts for data visualization
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios for API calls

### Backend
- **API Framework**: FastAPI with async support
- **Database**: PostgreSQL with psycopg2
- **AI Integration**: LangChain with OpenAI GPT-4
- **Vector Database**: ChromaDB for document retrieval
- **Streaming**: Server-Sent Events (SSE) for chat

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- PostgreSQL database with market sentiment data
- OpenAI API Key (optional, for chat functionality)

### 1. Environment Setup

Create a `.env` file in the root directory:

```env
# Database Configuration
DB_HOST=localhost
DB_NAME=market_sentiment
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install fastapi uvicorn psycopg2-binary pandas python-dotenv
pip install langchain langchain-openai langchain-community

# Or use the automated setup script
python run_backend.py
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Or if you prefer yarn
yarn install
```

### 4. Database Schema

Ensure your PostgreSQL database has the following tables:

- `commodities`: Commodity information
- `raw_news`: News articles
- `news_analysis_results`: Sentiment analysis results
- `daily_market_summary`: Daily aggregated sentiment data

## 🚀 Running the Application

### Start Backend Server

```bash
# Option 1: Using the run script (recommended)
python run_backend.py

# Option 2: Direct uvicorn command
uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API Endpoints: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

The web application will be available at: http://localhost:3000

## 📊 API Endpoints

### Dashboard APIs
- `GET /api/dashboard/sentiment-cards` - Get sentiment scores for all commodities
- `GET /api/dashboard/time-series/{commodity_name}` - Get time series data for a commodity
- `GET /api/dashboard/news/{commodity_name}` - Get news articles for a commodity
- `GET /api/dashboard/trending-keywords` - Get trending keywords

### Chat API
- `POST /api/chat` - Stream chat responses using Server-Sent Events

### Utility APIs
- `GET /health` - Health check endpoint

## 🎨 UI/UX Features

### Loading States
- Skeleton UI for sentiment cards during initial load
- Chart loading indicators
- News list loading placeholders
- Chat typing indicators

### Error Handling
- Connection error messages
- API failure notifications
- Empty state displays
- Retry mechanisms

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimizations
- Flexible grid layouts
- Touch-friendly interfaces

## 🔧 Configuration

### API Base URL
Update the `API_BASE_URL` constant in `frontend/src/app/page.tsx` if running backend on a different host/port.

### Sentiment Score Colors
- Green (≥60): Positive sentiment
- Orange (40-59): Neutral sentiment
- Red (≤40): Negative sentiment

### Chart Configuration
- Dual Y-axis: Sentiment (0-100) on left, Price ($) on right
- Time range: Based on available sentiment data
- Auto-refresh capability

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check database credentials in `.env`
   - Ensure database contains required tables and data

2. **CORS Issues**
   - Backend includes CORS middleware for localhost:3000
   - Update CORS origins if running frontend on different port

3. **Chat Not Working**
   - Verify OPENAI_API_KEY is set
   - Check OpenAI API quota and limits
   - Monitor backend logs for AI agent initialization

4. **Empty Data Display**
   - Verify database contains sample data
   - Check network connectivity between frontend and backend
   - Review browser console for API errors

### Development Tips
- Use browser dev tools Network tab to debug API calls
- Check backend logs for detailed error messages
- Verify database queries return expected data structure
- Test API endpoints directly using the Swagger docs at `/docs`

## 📝 Code Structure

### Frontend (`frontend/src/app/page.tsx`)
- Single-page application with all components
- Comprehensive state management
- TypeScript interfaces for type safety
- Material-UI component integration
- Responsive grid layouts

### Backend (`backend_api.py`)
- FastAPI application with CORS support
- PostgreSQL integration with connection pooling
- Pydantic models for API validation
- AI agent integration for chat functionality
- Error handling and logging

## 🚀 Deployment

### Production Considerations
- Set up environment variables securely
- Configure database connection pooling
- Enable HTTPS for production
- Set up reverse proxy (nginx)
- Configure logging and monitoring
- Implement rate limiting for API endpoints

### Docker Support (Optional)
Create Dockerfile for containerized deployment:
- Multi-stage build for frontend
- Python container for backend
- PostgreSQL container for database
- Docker Compose for orchestration

## 📄 License

This project is part of a market sentiment analysis system for agricultural commodities. Ensure compliance with data usage and API terms of service.

## 🤝 Contributing

1. Follow TypeScript and Python coding standards
2. Add comprehensive error handling
3. Include loading states for better UX
4. Write clear API documentation
5. Test responsive design on multiple devices
6. Validate database queries for performance