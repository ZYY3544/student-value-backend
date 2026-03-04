/**
 * ===========================================
 * ChatWidget - 简历优化助手浮窗组件
 * ===========================================
 *
 * 使用方式:
 *   import { ChatWidget } from './ChatWidget';
 *
 *   <ChatWidget
 *     assessmentContext={{ factors, abilities, grade, salaryRange, jobTitle, jobFunction }}
 *     resumeText={resumeText}
 *     apiBase="https://student-value-backend.onrender.com"
 *   />
 *
 * 放在 ResultView 的最外层 div 内即可。
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MessageCircle, X, Send, Loader2 } from 'lucide-react';

// ===========================================
// Types
// ===========================================

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatWidgetProps {
  assessmentContext: {
    factors: Record<string, string>;
    abilities: Record<string, any>;
    grade: number | null;
    salaryRange: string;
    jobTitle: string;
    jobFunction: string;
    deepInsight?: string;
  };
  resumeText: string;
  apiBase?: string;
}

// ===========================================
// Component
// ===========================================

export const ChatWidget: React.FC<ChatWidgetProps> = ({
  assessmentContext,
  resumeText,
  apiBase = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && !isInitializing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen, isInitializing]);

  // Initialize chat session when first opened
  const initSession = useCallback(async () => {
    if (sessionId) return; // Already initialized

    setIsInitializing(true);
    setError(null);

    try {
      const res = await fetch(`${apiBase}/api/chat/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assessmentContext,
          resumeText,
        }),
      });

      const data = await res.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to start chat');
      }

      setSessionId(data.data.sessionId);
      setMessages([{ role: 'assistant', content: data.data.greeting }]);
    } catch (err: any) {
      console.error('Chat init failed:', err);
      setError(err.message || 'Failed to connect');
    } finally {
      setIsInitializing(false);
    }
  }, [apiBase, assessmentContext, resumeText, sessionId]);

  // Handle opening chat
  const handleOpen = useCallback(() => {
    setIsOpen(true);
    if (!sessionId) {
      initSession();
    }
  }, [sessionId, initSession]);

  // Send message with SSE streaming
  const sendMessage = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isLoading || !sessionId) return;

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInputValue('');
    setIsLoading(true);

    // Add empty assistant message for streaming
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const res = await fetch(`${apiBase}/api/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          message: text,
          stream: true,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }

      // Read SSE stream
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === 'text') {
              fullText += event.content;
              // Update last message
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: fullText };
                return updated;
              });
            }
          } catch {
            // Skip invalid JSON
          }
        }
      }
    } catch (err: any) {
      console.error('Send failed:', err);
      // Update last message with error
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'Sorry, failed to get response. Please try again.',
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  }, [apiBase, inputValue, isLoading, sessionId]);

  // Handle keyboard
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  // Format text with **bold**
  const formatContent = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <span key={i} className="font-bold text-[#0A66C2]">
            {part.slice(2, -2)}
          </span>
        );
      }
      return part;
    });
  };

  return (
    <>
      {/* Floating Button */}
      {!isOpen && (
        <button
          onClick={handleOpen}
          className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-[#0A66C2] text-white shadow-lg flex items-center justify-center z-50 hover:bg-[#004182] active:scale-95 transition-all"
          style={{ boxShadow: '0 4px 20px rgba(10, 102, 194, 0.4)' }}
        >
          <MessageCircle size={26} />
        </button>
      )}

      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex flex-col bg-white sm:inset-auto sm:bottom-4 sm:right-4 sm:w-[400px] sm:h-[600px] sm:rounded-2xl sm:shadow-2xl sm:border sm:border-gray-200 overflow-hidden">
          {/* Header */}
          <div className="bg-[#0A66C2] text-white px-4 py-3 flex items-center gap-3 flex-shrink-0">
            <div className="w-9 h-9 rounded-full bg-white/20 flex items-center justify-center">
              <MessageCircle size={18} />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-sm font-bold">简历优化助手</h3>
              <p className="text-[11px] text-white/70">基于 HAY 评估体系</p>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="w-8 h-8 rounded-full hover:bg-white/20 flex items-center justify-center transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 bg-gray-50">
            {isInitializing && (
              <div className="flex items-center gap-2 text-gray-400 text-sm justify-center py-8">
                <Loader2 size={16} className="animate-spin" />
                <span>正在分析你的简历...</span>
              </div>
            )}

            {error && (
              <div className="text-center py-8">
                <p className="text-red-500 text-sm mb-2">{error}</p>
                <button
                  onClick={() => { setError(null); setSessionId(null); initSession(); }}
                  className="text-[#0A66C2] text-sm font-medium underline"
                >
                  重试
                </button>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                {/* Avatar */}
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-[11px] font-bold ${
                    msg.role === 'assistant'
                      ? 'bg-[#0A66C2] text-white'
                      : 'bg-gray-200 text-gray-500'
                  }`}
                >
                  {msg.role === 'assistant' ? 'AI' : 'Me'}
                </div>

                {/* Bubble */}
                <div
                  className={`max-w-[80%] px-3.5 py-2.5 text-[13px] leading-relaxed whitespace-pre-wrap break-words ${
                    msg.role === 'user'
                      ? 'bg-[#0A66C2] text-white rounded-2xl rounded-br-md'
                      : 'bg-white text-gray-800 rounded-2xl rounded-bl-md border border-gray-100 shadow-sm'
                  }`}
                >
                  {msg.role === 'assistant' ? (
                    msg.content ? (
                      formatContent(msg.content)
                    ) : (
                      <span className="flex items-center gap-1 text-gray-400">
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-300 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-300 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-300 animate-bounce" style={{ animationDelay: '300ms' }} />
                      </span>
                    )
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 bg-white px-3 py-2.5 flex items-end gap-2 flex-shrink-0">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题..."
              rows={1}
              disabled={isLoading || isInitializing}
              className="flex-1 resize-none border border-gray-200 rounded-xl px-3 py-2 text-[13px] outline-none focus:border-[#0A66C2] transition-colors disabled:bg-gray-50 disabled:text-gray-400"
              style={{ maxHeight: '80px' }}
              onInput={e => {
                const el = e.currentTarget;
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 80) + 'px';
              }}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading || isInitializing}
              className="w-9 h-9 rounded-xl bg-[#0A66C2] text-white flex items-center justify-center flex-shrink-0 disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-[#004182] active:scale-95 transition-all"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
        </div>
      )}
    </>
  );
};
