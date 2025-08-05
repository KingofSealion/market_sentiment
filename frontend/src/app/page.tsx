'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Typography,
  Tabs,
  Tab,
  Box,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemText,
  Paper,
  Chip,
  TextField,
  Button,
  IconButton,
  Skeleton,
  Alert,
  LinearProgress,
  Avatar,
  CardActionArea,
} from '@mui/material';
import {
  Send as SendIcon,
  Stop as StopIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import axios, { AxiosError } from 'axios';
import { TypingIndicator } from './components/TypingIndicator';

// API Base URL
const API_BASE_URL = 'http://localhost:8001';

// Types
interface SentimentCard {
  commodity_name: string;
  sentiment_score: number;
  reasoning: string;
  keywords: string[];
  last_updated: string;
}

interface TimeSeriesData {
  date: string;
  sentiment_score: number | null;
  price: number | null;
}

interface NewsArticle {
  id: number;
  title: string;
  sentiment_score: number;
  reasoning: string;
  keywords: string[];
  published_time: string;
  source: string;
}

interface TrendingKeyword {
  keyword: string;
  frequency: number;
}

interface ChatMessage {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: Date;
}


// Utility Functions
const getSentimentLevel = (score: number): string => {
  if (score >= 60) return 'Positive';
  if (score <= 40) return 'Negative';
  return 'Neutral';
};

const getSentimentColor = (score: number): string => {
  if (score >= 60) return '#4caf50'; // Green
  if (score <= 40) return '#f44336'; // Red
  return '#ff9800'; // Orange
};

const getSentimentBackgroundColor = (score: number): string => {
  if (score >= 60) return '#e8f5e8'; // Light green
  if (score <= 40) return '#ffebee'; // Light red
  return '#fff3e0'; // Light orange
};

const getCommodityIcon = (commodityName: string) => {
  const name = commodityName.toLowerCase();
  if (name.includes('corn')) return '🌽';
  if (name.includes('wheat')) return '🌾';
  if (name.includes('soybean meal')) return '🌱';
  if (name.includes('soybean oil')) return '🛢️';
  if (name.includes('soybean')) return '🌱';
  if (name.includes('palm')) return '🌴';
  return '🌾';
};

const getSentimentEmoji = (score: number): string => {
  if (score >= 60) return '🙂'; // Positive
  if (score <= 40) return '🙁'; // Negative
  return '😐'; // Neutral
};

const getPriceUnit = (commodityName: string): string => {
  const name = commodityName.toLowerCase();
  if (name.includes('corn') || name.includes('wheat') || name.includes('soybean')) {
    return 'Price (USc/bu)';
  }
  if (name.includes('soybean meal')) {
    return 'Price (USD/ST)';
  }
  if (name.includes('soybean oil')) {
    return 'Price (USc/lb)';
  }
  if (name.includes('palm oil')) {
    return 'Price (MYR/MT)';
  }
  return 'Price';
};

const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
};

const formatMarkdownText = (text: string): string => {
  if (!text) return '';
  
  return text
    // 줄바꿈 처리
    .replace(/\n/g, '<br/>')
    // **볼드** 처리
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // *이탤릭* 처리
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    // `인라인 코드` 처리
    .replace(/`(.*?)`/g, '<code style="background-color: #f1f5f9; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 0.9em;">$1</code>')
    // ### 제목 처리
    .replace(/^### (.*$)/gm, '<h3 style="font-size: 1.1em; font-weight: bold; margin: 12px 0 8px 0; color: #374151;">$1</h3>')
    // ## 제목 처리
    .replace(/^## (.*$)/gm, '<h2 style="font-size: 1.2em; font-weight: bold; margin: 16px 0 10px 0; color: #374151;">$1</h2>')
    // # 제목 처리
    .replace(/^# (.*$)/gm, '<h1 style="font-size: 1.3em; font-weight: bold; margin: 18px 0 12px 0; color: #374151;">$1</h1>')
    // --- 구분선 처리
    .replace(/^---$/gm, '<hr style="border: none; border-top: 1px solid #e5e7eb; margin: 16px 0;" />')
    // - 리스트 처리 (간단한 버전)
    .replace(/^- (.*$)/gm, '<div style="margin: 4px 0; padding-left: 16px;">• $1</div>')
    // 숫자 리스트 처리
    .replace(/^\d+\. (.*$)/gm, '<div style="margin: 4px 0; padding-left: 16px;">$&</div>');
};


// Main Component
export default function AgriCommoditiesDashboard() {
  // State Management
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedCommodity, setSelectedCommodity] = useState<string>('');
  const [sentimentCards, setSentimentCards] = useState<SentimentCard[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([]);
  const [trendingKeywords, setTrendingKeywords] = useState<TrendingKeyword[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  
  // console.log("1232323213", JSON.stringify(chatMessages, null, 2));
  // console.log("123213123", chatMessages) 
  
  // Loading States
  const [isLoadingCards, setIsLoadingCards] = useState(true);
  const [isLoadingChart, setIsLoadingChart] = useState(false);
  const [isLoadingNews, setIsLoadingNews] = useState(false);
  const [isLoadingKeywords, setIsLoadingKeywords] = useState(true);
  
  // Error States
  const [error, setError] = useState<string | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  // API Functions
  const fetchSentimentCards = async () => {
    try {
      setIsLoadingCards(true);
      setError(null);
      const response = await axios.get(`${API_BASE_URL}/api/dashboard/sentiment-cards`);
      setSentimentCards(response.data);
      if (response.data.length > 0 && !selectedCommodity) {
        setSelectedCommodity(response.data[0].commodity_name);
      }
    } catch (error) {
      const axiosError = error as AxiosError;
      setError(`데이터를 불러오는 데 실패했습니다: ${axiosError.message}`);
      console.error('Error fetching sentiment cards:', error);
    } finally {
      setIsLoadingCards(false);
    }
  };

  const fetchTimeSeriesData = async (commodity: string) => {
    if (!commodity) return;
    try {
      setIsLoadingChart(true);
      setError(null);
      const response = await axios.get(`${API_BASE_URL}/api/dashboard/time-series/${encodeURIComponent(commodity)}`);
      setTimeSeriesData(response.data);
    } catch (error) {
      const axiosError = error as AxiosError;
      if (axiosError.response?.status === 404) {
        setError(`Price data not available for ${commodity}. Please check if price_history table contains data for this commodity.`);
        setTimeSeriesData([]);
      } else {
        setError(`Failed to fetch price data for ${commodity}: ${axiosError.message}`);
        setTimeSeriesData([]);
      }
      console.error('Error fetching time series data:', error);
    } finally {
      setIsLoadingChart(false);
    }
  };

  const fetchNewsArticles = async (commodity: string) => {
    if (!commodity) return;
    try {
      setIsLoadingNews(true);
      const response = await axios.get(`${API_BASE_URL}/api/dashboard/news/${encodeURIComponent(commodity)}`);
      setNewsArticles(response.data);
    } catch (error) {
      console.error('Error fetching news articles:', error);
    } finally {
      setIsLoadingNews(false);
    }
  };

  const fetchTrendingKeywords = async () => {
    try {
      setIsLoadingKeywords(true);
      const response = await axios.get(`${API_BASE_URL}/api/dashboard/trending-keywords`);
      setTrendingKeywords(response.data);
    } catch (error) {
      console.error('Error fetching trending keywords:', error);
    } finally {
      setIsLoadingKeywords(false);
    }
  };

  // Chat Functions
  const sendChatMessage = async () => {
    if (!chatInput.trim() || isChatLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: chatInput.trim(),
      isUser: true,
      timestamp: new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    // Create new AbortController for this request
    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage.message }),
        signal: controller.signal, // Add abort signal
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      let botMessageContent = '';
      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: '',
        isUser: false,
        timestamp: new Date(),
      };

      setChatMessages(prev => [...prev, botMessage]);

      try {
        while (true) {
          // Check if request was aborted
          if (controller.signal.aborted) {
            throw new Error('Request was cancelled');
          }

          const { done, value } = await reader.read();
          if (done) break;

          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'chunk' || data.type === 'end') {
                  botMessageContent = data.message;
                  setChatMessages(prev =>
                    prev.map(msg =>
                      msg.id === botMessage.id
                        ? { ...msg, message: botMessageContent }
                        : msg
                    )
                  );
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error: any) {
      console.error('Error sending chat message:', error);
      
      // Handle different error types
      let errorMessage = '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.';
      if (error.name === 'AbortError' || error.message === 'Request was cancelled') {
        errorMessage = '요청이 취소되었습니다.';
      }
      
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: errorMessage,
        isUser: false,
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsChatLoading(false);
      setAbortController(null);
    }
  };

  const stopChatMessage = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsChatLoading(false);
    }
  };

  const handleNewConversation = () => {
    setChatMessages([]);
  };

  // Effects
  useEffect(() => {
    fetchSentimentCards();
    fetchTrendingKeywords();
  }, []);

  useEffect(() => {
    if (selectedCommodity) {
      fetchTimeSeriesData(selectedCommodity);
      fetchNewsArticles(selectedCommodity);
    }
  }, [selectedCommodity]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Render Functions
  const renderMarketOverview = () => {
    // Define commodity order as requested
    const commodityOrder = ['Corn', 'Wheat', 'Soybean', 'Soybean Meal', 'Soybean Oil', 'Palm Oil'];
    
    // Sort cards according to the specified order
    const sortedCards = commodityOrder.map(commodity => 
      sentimentCards.find(card => card.commodity_name === commodity)
    ).filter(Boolean);

    if (isLoadingCards) {
      return (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold', color: 'text.primary' }}>
            📊 Market Overview
          </Typography>
          <Grid 
            container 
            spacing={1}
            justifyContent="flex-start"
            alignItems="stretch"
            sx={{ width: '100%' }}
          >
            {[...Array(6)].map((_, index) => (
              <Grid item xs={12} sm={6} md={4} lg={2} xl={2} key={index}>
                <Card sx={{ 
                  height: 200,
                  minWidth: 0,
                  maxWidth: '100%',
                  overflow: 'hidden',
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease-in-out',
                }}>
                  <Skeleton variant="rectangular" height={200} />
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      );
    }

    if (sentimentCards.length === 0) {
      return (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold', color: 'text.primary' }}>
            📊 Market Overview
          </Typography>
          <Alert severity="info">
            감정점수 데이터가 없습니다. 데이터베이스 연결을 확인해주세요.
          </Alert>
        </Box>
      );
    }

    return (
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold', color: 'text.primary' }}>
          📊 Market Overview
        </Typography>
        <Grid 
          container 
          spacing={2}
          justifyContent="space-between"
          alignItems="stretch"
          sx={{ width: '100%' }}
        >
          {sortedCards.map((card) => (
            <Grid item xs={12} sm={6} md={4} lg key={card.commodity_name} sx={{ flex: 1, display: 'flex' }}>
              <Card
                sx={{
                  height: 240,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease-in-out',
                  backgroundColor: selectedCommodity === card.commodity_name 
                    ? getSentimentBackgroundColor(card.sentiment_score)
                    : 'background.paper',
                  border: selectedCommodity === card.commodity_name 
                    ? `2px solid ${getSentimentColor(card.sentiment_score)}`
                    : '1px solid',
                  borderColor: selectedCommodity === card.commodity_name 
                    ? getSentimentColor(card.sentiment_score)
                    : 'divider',
                  boxShadow: selectedCommodity === card.commodity_name ? 3 : 1,
                  '&:hover': {
                    boxShadow: 3,
                    transform: 'translateY(-2px)',
                  },
                  // 페이지 전체 가로에 맞게 카드 크기 최대화
                  minWidth: 0,
                  maxWidth: '100%',
                  width: '100%',
                  overflow: 'hidden',
                  boxSizing: 'border-box',
                  flex: 1,
                }}
                onClick={() => setSelectedCommodity(card.commodity_name)}
              >
                <CardActionArea sx={{ height: '100%', p: 0 }}>
                  <CardContent sx={{ 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column',
                    p: 2, // 패딩 증가
                    '&:last-child': { pb: 2 },
                    minWidth: 0,
                    overflow: 'hidden',
                    width: '100%',
                  }}>
                    {/* Commodity name and icon */}
                    <Box sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      mb: 1.5,
                      minWidth: 0,
                      overflow: 'hidden',
                    }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: getSentimentColor(card.sentiment_score),
                          width: 36, // 크기 증가
                          height: 36,
                          borderRadius: '50%',
                          mr: 1.5,
                          fontSize: '1.2rem', // 폰트 크기 증가
                          flexShrink: 0,
                        }}
                      >
                        {getCommodityIcon(card.commodity_name)}
                      </Box>
                      <Typography 
                        variant="subtitle1" 
                        sx={{ 
                          fontWeight: 'bold', 
                          fontSize: '0.95rem', // 폰트 크기 증가
                          minWidth: 0,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {card.commodity_name}
                      </Typography>
                    </Box>

                    {/* Sentiment score and level */}
                    <Box sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center', 
                      mb: 1.5,
                      minWidth: 0,
                    }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 50, // 크기 증가
                          height: 50,
                          borderRadius: '50%',
                          bgcolor: getSentimentColor(card.sentiment_score),
                          color: 'white',
                          mr: 1.5,
                          flexShrink: 0,
                        }}
                      >
                        <Typography variant="h6" sx={{ fontWeight: 'bold', fontSize: '1.1rem' }}>
                          {Math.round(card.sentiment_score)}
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center', minWidth: 0 }}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            color: getSentimentColor(card.sentiment_score),
                            fontWeight: 'bold',
                            display: 'block',
                            fontSize: '0.8rem',
                          }}
                        >
                          {getSentimentLevel(card.sentiment_score)}
                        </Typography>
                        {card.sentiment_score >= 60 ? (
                          <TrendingUpIcon sx={{ color: getSentimentColor(card.sentiment_score), fontSize: 20 }} />
                        ) : card.sentiment_score <= 40 ? (
                          <TrendingDownIcon sx={{ color: getSentimentColor(card.sentiment_score), fontSize: 20 }} />
                        ) : (
                          <TrendingUpIcon sx={{ color: getSentimentColor(card.sentiment_score), fontSize: 20 }} />
                        )}
                      </Box>
                    </Box>

                    {/* Keywords */}
                    <Box sx={{ 
                      display: 'flex', 
                      flexWrap: 'wrap', 
                      flexDirection: 'column',   // 👈 가로 정렬을 세로로 변경
                      gap: 1, 
                      mb: 1.5, 
                      minHeight: 32,
                      overflow: 'hidden',
                      minWidth: 0,
                    }}>
                      {card.keywords.slice(0, 2).map((keyword, index) => (
                        <Chip
                          key={index}
                          label={keyword}
                          size="small"
                          variant="outlined"
                          sx={{ 
                            fontSize: '0.65rem', // 크기 증가
                            height: 22, // 높이 증가
                            minWidth: 0,
                            '& .MuiChip-label': {
                              px: 0.7,
                            },
                          }}
                        />
                      ))}
                    </Box>

                    {/* Last updated */}
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        color: 'text.secondary', 
                        mt: 'auto',
                        textAlign: 'center',
                        fontSize: '0.75rem', // 크기 증가
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {formatDate(card.last_updated)}
                    </Typography>
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  // Render Sentiment Analysis Section
  const renderSentimentAnalysis = () => {
    // Get selected commodity data
    const selectedCard = sentimentCards.find(card => card.commodity_name === selectedCommodity);

    if (isLoadingCards) {
      return (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Skeleton variant="text" width="40%" height={32} />
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Skeleton variant="circular" width={60} height={60} />
                <Box sx={{ flex: 1 }}>
                  <Skeleton variant="text" width="30%" height={40} />
                  <Skeleton variant="text" width="20%" height={24} />
                </Box>
              </Box>
              <Skeleton variant="text" width="100%" height={20} />
              <Skeleton variant="text" width="100%" height={20} />
              <Skeleton variant="text" width="80%" height={20} />
              <Skeleton variant="text" width="60%" height={20} />
            </Box>
          </CardContent>
        </Card>
      );
    }

    if (!selectedCard) {
      return (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 'bold' }}>
              📊 Sentiment Analysis
            </Typography>
            <Alert severity="info">
              Please select a commodity from the Market Overview to view detailed sentiment analysis.
            </Alert>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card sx={{ 
        height: 'fit-content', 
        maxHeight: '500px',
        width: '100%',
        minWidth: 0, // Critical: prevents card from expanding beyond container
        overflow: 'hidden', // Prevents any overflow
      }}>
        <CardContent sx={{ 
          width: '100%',
          minWidth: 0, // Critical: allows content to shrink
          overflow: 'hidden',
        }}>
          <Typography 
            variant="h6" 
            sx={{ 
              mb: 2, 
              fontWeight: 'bold',
              width: '100%',
              minWidth: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            📊 Sentiment Analysis - {selectedCard.commodity_name}
          </Typography>
          
          {/* Main sentiment display */}
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 2, 
            mb: 3, 
            p: 2,
            bgcolor: getSentimentBackgroundColor(selectedCard.sentiment_score),
            borderRadius: 2,
            border: `2px solid ${getSentimentColor(selectedCard.sentiment_score)}`,
            width: '100%',
            minWidth: 0,
            maxWidth: '100%',
            overflow: 'hidden',
          }}>
            {/* Sentiment Emoji */}
            <Box sx={{ 
              fontSize: '3.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {getSentimentEmoji(selectedCard.sentiment_score)}
            </Box>
            
            {/* Score and Level */}
            <Box sx={{ flex: 1 }}>
              <Typography 
                variant="h3" 
                sx={{ 
                  fontWeight: 'bold',
                  color: getSentimentColor(selectedCard.sentiment_score),
                  mb: 1,
                }}
              >
                {Math.round(selectedCard.sentiment_score)}
              </Typography>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 'bold',
                  color: getSentimentColor(selectedCard.sentiment_score),
                }}
              >
                {getSentimentLevel(selectedCard.sentiment_score)}
              </Typography>
            </Box>
          </Box>

          {/* Reasoning */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: 'text.primary' }}>
              💭 Analysis Reasoning
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                color: 'text.secondary',
                lineHeight: 1.4,
                p: 2,
                bgcolor: 'grey.50',
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'grey.200',
                
                // Critical: Complete text overflow control
                width: '100%',
                minWidth: 0,
                maxWidth: '100%',
                wordBreak: 'break-word',
                overflowWrap: 'break-word',
                whiteSpace: 'normal',
                overflow: 'hidden',
                
                // Force 3-line limit
                display: '-webkit-box',
                WebkitLineClamp: 3,
                WebkitBoxOrient: 'vertical',
                textOverflow: 'ellipsis',
                
                // Fixed height prevents expansion
                height: '4.2em', // Fixed height instead of maxHeight
                
                // Prevent any layout shifts
                boxSizing: 'border-box',
              }}
            >
              {selectedCard.reasoning}
            </Typography>
          </Box>

          {/* Keywords */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: 'text.primary' }}>
              🏷️ Key Factors
            </Typography>
            <Box sx={{ 
              display: 'flex', 
              flexWrap: 'wrap', 
              gap: 0.5,
              maxHeight: '60px',
              overflow: 'hidden'
            }}>
              {selectedCard.keywords.slice(0, 6).map((keyword, index) => (
                <Chip
                  key={index}
                  label={keyword}
                  variant="outlined"
                  size="small"
                  sx={{
                    borderColor: getSentimentColor(selectedCard.sentiment_score),
                    color: getSentimentColor(selectedCard.sentiment_score),
                    fontWeight: 'medium',
                    fontSize: '0.75rem',
                    height: '24px',
                    '&:hover': {
                      bgcolor: getSentimentBackgroundColor(selectedCard.sentiment_score),
                    },
                  }}
                />
              ))}
              {selectedCard.keywords.length > 6 && (
                <Chip
                  label={`+${selectedCard.keywords.length - 6} more`}
                  variant="outlined"
                  size="small"
                  sx={{
                    borderColor: 'grey.400',
                    color: 'grey.600',
                    fontSize: '0.7rem',
                    height: '24px',
                  }}
                />
              )}
            </Box>
          </Box>

          {/* Last Updated */}
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            pt: 2,
            borderTop: '1px solid',
            borderColor: 'grey.200',
          }}>
            <Typography variant="caption" color="text.secondary">
              📅 Last Updated: {formatDate(selectedCard.last_updated)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              🌾 {selectedCard.commodity_name}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  };

  // Render Sentiment Trends Chart Section
  const renderSentimentTrendsChart = () => {
    const priceUnit = selectedCommodity ? getPriceUnit(selectedCommodity) : 'Price';

    if (isLoadingChart) {
      return (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Skeleton variant="text" width="40%" height={32} />
              <Skeleton variant="rectangular" height={360} />
            </Box>
          </CardContent>
        </Card>
      );
    }

    if (!selectedCommodity) {
      return (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 'bold' }}>
              📈 Sentiment Trends
            </Typography>
            <Alert severity="info">
              Please select a commodity from the Market Overview to view sentiment trends.
            </Alert>
          </CardContent>
        </Card>
      );
    }

    if (timeSeriesData.length === 0 && !isLoadingChart) {
      return (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 'bold' }}>
              📈 Sentiment Trends - {selectedCommodity}
            </Typography>
            <Alert severity="error" sx={{ mb: 2 }}>
              <strong>Price data not available for {selectedCommodity}</strong>
            </Alert>
            <Alert severity="info">
              This chart requires data from the <code>price_history</code> table in PostgreSQL. 
              Please ensure the table exists and contains price data for {selectedCommodity}.
            </Alert>
          </CardContent>
        </Card>
      );
    }

    // Calculate dynamic scaling for both axes
    const priceValues = timeSeriesData
      .map(item => item.price)
      .filter(price => price !== null && price !== undefined) as number[];
    
    const sentimentValues = timeSeriesData
      .map(item => item.sentiment_score)
      .filter(score => score !== null && score !== undefined) as number[];
    
    // Dynamic Price Axis Scaling
    const minPrice = Math.min(...priceValues);
    const maxPrice = Math.max(...priceValues);
    let dynamicMinPrice, dynamicMaxPrice;
    
    // 데이터가 하나뿐이거나 모든 값이 같을 경우를 대비
    if (minPrice === maxPrice) {
      // 값의 5%만큼 위아래 여백을 줍니다.
      dynamicMinPrice = minPrice * 0.95;
      dynamicMaxPrice = maxPrice * 1.05;
    } else {
      // 실제 데이터의 변동폭(range)을 계산
      const dataRange = maxPrice - minPrice;
      // 변동폭의 10%를 위아래 여백(padding)으로 설정
      const padding = dataRange * 0.1;
    
      // 최소값에서 여백을 빼고, 최대값에 여백을 더해 새로운 범위를 설정
      // Math.floor/ceil을 사용해 축의 숫자를 깔끔하게 만듭니다.
      dynamicMinPrice = Math.floor(minPrice - padding);
      dynamicMaxPrice = Math.ceil(maxPrice + padding);
    }
    
    // Dynamic Sentiment Score Axis Scaling (±5 points)
    const minSentiment = Math.min(...sentimentValues);
    const maxSentiment = Math.max(...sentimentValues);
    let dynamicMinSentiment, dynamicMaxSentiment;
    
    // 모든 점수가 같을 경우를 대비
    if (minSentiment === maxSentiment) {
      // 위아래로 5점씩 여백을 주되, 0~100 범위를 벗어나지 않게 함
      dynamicMinSentiment = Math.max(0, minSentiment - 5);
      dynamicMaxSentiment = Math.min(100, maxSentiment + 5);
    } else {
      // 실제 데이터의 변동폭(range)을 계산
      const dataRange = maxSentiment - minSentiment;
      // 변동폭의 10%를 위아래 여백(padding)으로 설정
      const padding = dataRange * 0.1;
    
      // 여백을 적용한 후, 결과값이 0~100 사이를 벗어나지 않도록 범위를 제한
      dynamicMinSentiment = Math.max(0, minSentiment - padding);
      dynamicMaxSentiment = Math.min(100, maxSentiment + padding);
    }

    return (
      <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 2, '&:last-child': { pb: 2 } }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              📈 Sentiment Trends - {selectedCommodity}
            </Typography>
            <IconButton
              onClick={() => fetchTimeSeriesData(selectedCommodity)}
              size="small"
              sx={{ color: 'text.secondary' }}
              title="Refresh data"
            >
              <RefreshIcon />
            </IconButton>
          </Box>
          
          <Box sx={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart 
                data={timeSeriesData} 
                margin={{ top: 5, right: 20, left: 20, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value) => formatDate(value)}
                  stroke="#666"
                  fontSize={11}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  yAxisId="sentiment" 
                  orientation="left" 
                  domain={[dynamicMinSentiment, dynamicMaxSentiment]}
                  stroke="#2196f3"
                  label={{ 
                    value: 'Sentiment Score', 
                    angle: -90, 
                    dx : -15,
                    style: { textAnchor: 'middle' }
                  }}
                  fontSize={11}
                  tickFormatter={(value) => value.toFixed(0)}
                />
                <YAxis 
                  yAxisId="price" 
                  orientation="right" 
                  domain={[dynamicMinPrice, dynamicMaxPrice]}
                  stroke="#ff9800"
                  label={{ 
                    value: priceUnit, 
                    angle: 90, 
                    dx: 30,
                    style: { textAnchor: 'middle' }
                  }}
                  fontSize={11}
                  tickFormatter={(value) => value.toFixed(2)}
                />
                <Tooltip
                  labelFormatter={(value) => `Date: ${formatDate(value)}`}
                  formatter={(value, name, props) => {
                    const formattedValue = typeof value === 'number' ? value.toFixed(2) : 'N/A';
                    
                    // Enhanced tooltip clarity
                    if (props.dataKey === 'sentiment_score') {
                      return [formattedValue, 'Sentiment Score'];
                    } else if (props.dataKey === 'price') {
                      return [formattedValue, priceUnit];
                    }
                    
                    return [formattedValue, name];
                  }}
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.98)',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                    fontSize: '13px'
                  }}
                  cursor={{ stroke: '#666', strokeWidth: 1, strokeDasharray: '3 3' }}
                />
                <Legend
                  verticalAlign="bottom"
                  iconType="line"
                  wrapperStyle={{
                    position: 'absolute',
                    bottom: 0,
                    left: '50%',
                    transform: 'translateX(-50%)'
                  }}
                />
                <Line
                  yAxisId="sentiment"
                  type="monotone"
                  dataKey="sentiment_score"
                  stroke="#2196f3"
                  strokeWidth={3}
                  name="Sentiment Score"
                  connectNulls={false}
                  dot={{ fill: '#2196f3', strokeWidth: 2, r: 3 }}
                  activeDot={{ 
                    r: 6, 
                    stroke: '#2196f3', 
                    strokeWidth: 2, 
                    fill: '#fff' 
                  }}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="price"
                  stroke="#ff9800"
                  strokeWidth={3}
                  name={priceUnit}
                  connectNulls={false}
                  dot={{ fill: '#ff9800', strokeWidth: 2, r: 3 }}
                  activeDot={{ 
                    r: 6, 
                    stroke: '#ff9800', 
                    strokeWidth: 2, 
                    fill: '#fff' 
                  }}
                />
                {/* Interactive Brush for zoom functionality */}
                {timeSeriesData.length > 10 && (
                  <defs>
                    <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8884d8" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                )}
              </LineChart>
            </ResponsiveContainer>
          </Box>
          
          {/* Interactive Controls */}
          {timeSeriesData.length > 0 && (
            <Box sx={{ 
              mt: 2, 
              p: 2, 
              bgcolor: 'grey.50', 
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2
            }}>
              <Typography variant="caption" color="text.secondary">
                💡 Chart Tips: Hover over data points for details • Use mouse wheel to zoom • Drag chart to pan
              </Typography>
            </Box>
          )}

          {/* Data source note */}
          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'grey.200' }}>
            <Typography variant="caption" color="text.secondary">
              📊 Sentiment data from daily market analysis • Price data from market feeds
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const renderNewsArticles = () => {
    if (isLoadingNews) {
      return (
        <Card sx={{ height: 1000, display: 'flex', flexDirection: 'column' }}>
          <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 3, '&:last-child': { pb: 3 } }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Skeleton variant="text" width="30%" height={32} />
              {[...Array(5)].map((_, index) => (
                <Box key={index} sx={{ pb: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
                  <Skeleton variant="text" width="80%" height={24} sx={{ mb: 1 }} />
                  <Skeleton variant="text" width="100%" height={20} sx={{ mb: 1 }} />
                  <Skeleton variant="text" width="60%" height={20} />
                </Box>
              ))}
            </Box>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card sx={{ height: 700, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 3, '&:last-child': { pb: 3 }, overflow: 'hidden' }}>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 'bold' }}>
            📰 최근 뉴스 & 분석 - {selectedCommodity}
          </Typography>
          
          {newsArticles.length === 0 ? (
            <Alert severity="info">
              표시할 기사가 없습니다.
            </Alert>
          ) : (
            <Box sx={{ 
              height: 580, // 고정 높이 설정 (700px - 타이틀 및 패딩 120px)
              overflowY: 'auto', // auto로 변경하여 필요할 때만 스크롤바 표시
              display: 'flex', 
              flexDirection: 'column', 
              gap: 2,
              pr: 1, // 스크롤바를 위한 오른쪽 패딩
              border: '1px solid #e0e0e0', // 디버그용 경계선
              borderRadius: 1,
              p: 1,
              '&::-webkit-scrollbar': {
                width: '8px',
              },
              '&::-webkit-scrollbar-track': {
                background: '#f5f5f5',
                borderRadius: '4px',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#bbb',
                borderRadius: '4px',
                '&:hover': {
                  background: '#999',
                },
              },
              // Firefox용 스크롤바
              scrollbarWidth: 'thin',
              scrollbarColor: '#bbb #f5f5f5',
            }}>
              {newsArticles.map((article, index) => (
                <Card
                  key={`${article.id}-${index}`}
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    minHeight:200,
                    transition: 'all 0.2s ease-in-out',
                    '&:hover': {
                      bgcolor: 'grey.50',
                      boxShadow: 2,
                    },
                  }}
                  onClick={() => alert(`기사 상세보기:\n\n제목: ${article.title}\n\n분석: ${article.reasoning}`)}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                      <Typography 
                        variant="subtitle2" 
                        sx={{ 
                          fontWeight: 'medium',
                          flex: 1,
                          pr: 2,
                          lineHeight: 1.3
                        }}
                      >
                        {article.title}
                      </Typography>
                      <Chip
                        label={Math.round(article.sentiment_score)}
                        size="small"
                        sx={{
                          bgcolor: getSentimentColor(article.sentiment_score),
                          color: 'white',
                          fontWeight: 'bold',
                          minWidth: 40,
                        }}
                      />
                    </Box>
                    
                    <Typography 
                      variant="body2" 
                      color="text.secondary" 
                      sx={{ 
                        mb: 2,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}
                    >
                      {article.reasoning}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                      {article.keywords.slice(0, 3).map((keyword, index) => (
                        <Chip
                          key={index}
                          label={keyword}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.7rem', height: 24 }}
                        />
                      ))}
                    </Box>
                    
                    <Typography variant="caption" color="text.secondary">
                      {article.source} • {formatDate(article.published_time)}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderTrendingKeywords = () => {
    if (isLoadingKeywords) {
      return (
        <Card>
          <CardContent>
            <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {[...Array(8)].map((_, index) => (
                <Skeleton key={index} variant="rounded" width={80} height={32} />
              ))}
            </Box>
          </CardContent>
        </Card>
      );
    }

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 'bold' }}>
            🔥 트렌딩 키워드
          </Typography>
          
          {trendingKeywords.length === 0 ? (
            <Alert severity="info">
              트렌딩 키워드가 없습니다.
            </Alert>
          ) : (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {trendingKeywords.map((keyword, index) => (
                <Chip
                  key={index}
                  label={`${keyword.keyword} (${keyword.frequency})`}
                  sx={{
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    fontWeight: 'medium',
                    '&:hover': {
                      bgcolor: 'primary.dark',
                    },
                    cursor: 'pointer',
                    transition: 'all 0.2s ease-in-out',
                  }}
                />
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderChatbot = () => (
    <div className="max-w-9xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">AI 원자재 시장 분석가</h2>
        <Button
          onClick={handleNewConversation}
          variant="outlined"
          size="small"
          className="text-gray-600 border-gray-300"
        >
          새 대화
        </Button>
      </div>

      <div className="bg-white rounded-lg shadow-md p-4 h-96 mb-4 overflow-y-auto">
        {chatMessages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            원자재 시장에 대해 무엇이든 물어보세요! 
          </div>
        ) : (
          <div className="space-y-4">
            {chatMessages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-4xl lg:max-w-7xl px-4 py-2 rounded-lg ${
                    message.isUser
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                 
                  <div 
                    className="text-sm whitespace-pre-wrap break-words leading-relaxed" 
                    // style={{ 
                    //   whiteSpace: 'pre-wrap',
                    //   wordBreak: 'break-word',
                    //   overflowWrap: 'break-word',
                    //   lineHeight: '1.6',
                    //   fontFamily: 'inherit'
                    // }}
                    dangerouslySetInnerHTML={{
                      __html: formatMarkdownText(message.message)
                    }}
                  />
                  <p className={`text-xs mt-1 ${message.isUser ? 'text-blue-100' : 'text-gray-500'}`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            {isChatLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg">
                  <TypingIndicator />
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="flex gap-3">
        <TextField
          fullWidth
          variant="outlined"
          placeholder="원자재 시장에 대해 질문하세요..."
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
          disabled={isChatLoading}
          size="small"
          className="bg-white"
        />
        <IconButton
          onClick={isChatLoading ? stopChatMessage : sendChatMessage}
          disabled={!isChatLoading && !chatInput.trim()}
          className={`text-white hover:bg-blue-600 disabled:bg-gray-300 ${
            isChatLoading ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500'
          }`}
          title={isChatLoading ? '응답 생성 중지' : '메시지 보내기'}
        >
          {isChatLoading ? <StopIcon /> : <SendIcon />}
        </IconButton>
      </div>
    </div>
  );

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'grey.50' }}>
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'text.primary', mb: 1 }}>
            🌾 원자재 시장 심리화 대시보드
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Real-Time Commodities Market News Sentiment Analysis & AI Insights
          </Typography>
        </Box>

        {error && (
          <Box sx={{ mb: 3 }}>
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          </Box>
        )}

        <Box sx={{ mb: 4 }}>
          <Paper sx={{ borderRadius: 2 }}>
            <Tabs
              value={currentTab}
              onChange={(_, newValue) => setCurrentTab(newValue)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{
                '& .MuiTab-root': {
                  fontWeight: 'bold',
                  fontSize: '1rem',
                },
              }}
            >
              <Tab label="📊 대시보드" />
              <Tab label="🤖 AI 챗봇" />
            </Tabs>
          </Paper>
        </Box>

        {currentTab === 0 && (
          <Box>
            {renderMarketOverview()}
            
            {/* New two-column layout below Market Overview - Fixed 50:50 Split */}
            <Box sx={{ 
              display: 'flex', 
              gap: 3, 
              mb: 4,
              flexDirection: { xs: 'column', md: 'row' },
              width: '100%'
            }}>
              <Box sx={{ 
                flex: '1 1 50%',
                minWidth: 0, // Critical: allows flex children to shrink below content size
                maxWidth: { xs: '100%', md: '50%' },
                width: { xs: '100%', md: '50%' },
              }}>
                {renderSentimentAnalysis()}
              </Box>
              
              <Box sx={{ 
                flex: '1 1 50%',
                minWidth: 0, // Critical: allows flex children to shrink below content size
                maxWidth: { xs: '100%', md: '50%' },
                width: { xs: '100%', md: '50%' },
              }}>
                {renderSentimentTrendsChart()}
              </Box>
            </Box>
            
            {/* Existing layout for news and trending keywords */}
            <Grid container spacing={2} alignItems="stretch">
              <Grid xs={12} lg={8}>
                {renderNewsArticles()}
              </Grid>
              
              <Grid xs={12} lg={4}>
                {renderTrendingKeywords()}
              </Grid>
            </Grid>
          </Box>
        )}

        {currentTab === 1 && renderChatbot()}
      </Container>
    </Box>
  );
}