'use client';

import React, { useState, useEffect } from 'react';

// Simple interfaces
interface SentimentCard {
  commodity_name: string;
  sentiment_score: number;
  reasoning: string;
  keywords: string[];
  last_updated: string;
}

// API Base URL
const API_BASE_URL = 'http://localhost:8001';

// Simple utility
const getSentimentColor = (score: number): string => {
  if (score >= 60) return '#4caf50'; // Green
  if (score <= 40) return '#f44336'; // Red
  return '#ff9800'; // Orange
};

export default function SimpleDashboard() {
  const [sentimentCards, setSentimentCards] = useState<SentimentCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch data function
  const fetchSentimentCards = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/dashboard/sentiment-cards`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setSentimentCards(data);
    } catch (error) {
      console.error('Error fetching sentiment cards:', error);
      setError(`Failed to load data: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSentimentCards();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h1>🌾 Agri Commodities Sentiment Dashboard</h1>
        <p>Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h1>🌾 Agri Commodities Sentiment Dashboard</h1>
        <div style={{ color: 'red', margin: '20px 0' }}>
          {error}
        </div>
        <button onClick={fetchSentimentCards}>Retry</button>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>
        🌾 Agri Commodities Sentiment Dashboard
      </h1>

      {sentimentCards.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <p>No sentiment data available. Please check your database connection.</p>
        </div>
      ) : (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '20px',
          marginBottom: '30px'
        }}>
          {sentimentCards.map((card) => (
            <div
              key={card.commodity_name}
              style={{
                border: '1px solid #ddd',
                borderRadius: '8px',
                padding: '20px',
                backgroundColor: '#fff',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                cursor: 'pointer',
                transition: 'box-shadow 0.2s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
              }}
            >
              <h3 style={{ margin: '0 0 15px 0', fontSize: '18px' }}>
                {card.commodity_name}
              </h3>
              
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
                <span
                  style={{
                    fontSize: '36px',
                    fontWeight: 'bold',
                    color: getSentimentColor(card.sentiment_score),
                    marginRight: '10px'
                  }}
                >
                  {card.sentiment_score.toFixed(0)}
                </span>
                <span style={{ fontSize: '18px' }}>
                  {card.sentiment_score >= 60 ? '📈' : card.sentiment_score <= 40 ? '📉' : '➡️'}
                </span>
              </div>

              <p style={{ 
                fontSize: '14px', 
                color: '#666', 
                marginBottom: '15px',
                lineHeight: '1.4'
              }}>
                {card.reasoning}
              </p>

              <div style={{ marginBottom: '15px' }}>
                {card.keywords.slice(0, 3).map((keyword, index) => (
                  <span
                    key={index}
                    style={{
                      display: 'inline-block',
                      padding: '4px 8px',
                      margin: '2px 4px 2px 0',
                      backgroundColor: '#f0f0f0',
                      borderRadius: '12px',
                      fontSize: '12px',
                      color: '#333'
                    }}
                  >
                    {keyword}
                  </span>
                ))}
              </div>

              <p style={{ fontSize: '12px', color: '#999', margin: 0 }}>
                Updated: {new Date(card.last_updated).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      )}

      <div style={{ textAlign: 'center', marginTop: '40px' }}>
        <h2>API Connection Test</h2>
        <p>✅ Successfully connected to backend API</p>
        <p>📊 Loaded {sentimentCards.length} commodity sentiment cards</p>
        <button 
          onClick={fetchSentimentCards}
          style={{
            padding: '10px 20px',
            backgroundColor: '#1976d2',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Refresh Data
        </button>
      </div>
    </div>
  );
}