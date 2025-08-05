'use client';

import React from 'react';

interface TypingIndicatorProps {
  message?: string;
  className?: string;
}

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({ 
  message = "AI가 입력 중", 
  className = "" 
}) => {
  const dotStyle = {
    width: '8px',
    height: '8px',
    backgroundColor: '#3b82f6',
    borderRadius: '50%',
    display: 'inline-block',
    animation: 'typing-bounce 1.4s ease-in-out infinite',
    transformOrigin: 'center bottom'
  };

  const dot2Style = {
    ...dotStyle,
    animationDelay: '0.2s'
  };

  const dot3Style = {
    ...dotStyle,
    animationDelay: '0.4s'
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }} className={className}>
      <span style={{ fontSize: '14px', color: '#6b7280' }}>{message}</span>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '16px' }}>
        <div style={dotStyle} />
        <div style={dot2Style} />
        <div style={dot3Style} />
      </div>
    </div>
  );
};

export default TypingIndicator;