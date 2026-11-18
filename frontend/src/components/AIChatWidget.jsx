/**
 * src/components/AIChatWidget.jsx
 * =================================
 * Floating AI Health Assistant (HealthBot) chat widget.
 * Powered by Grok (xAI) via /api/chat backend endpoint.
 *
 * Features:
 * - Floating action button (bottom-right, z-50)
 * - Slide-up chat panel with message history
 * - User messages (right, blue) + HealthBot messages (left, slate)
 * - Typing indicator animation while waiting
 * - Persistent session messages (cleared on unmount)
 * - Disclaimer banner
 * - Quick starter prompts
 */

import { useState, useRef, useEffect } from 'react'
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  AlertCircle,
  Loader2,
  Sparkles,
} from 'lucide-react'
import { sendChatMessage } from '../services/api'

const QUICK_PROMPTS = [
  'What does my risk score mean?',
  'How does BMI affect diabetes risk?',
  'What is prediabetes?',
  'How can I improve my health score?',
]

const WELCOME_MESSAGE = {
  role: 'assistant',
  content:
    "👋 Hi! I'm **HealthBot**, your AI health education assistant.\n\nI can help you understand your assessment results, explain health indicators, and answer general questions about diabetes risk. How can I help you today?",
}

function formatMessage(text) {
  // Simple markdown-like formatting for bold and line breaks
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br />')
}

export default function AIChatWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Scroll to bottom when messages update
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isOpen, loading])

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 150)
    }
  }, [isOpen])

  const handleSend = async (text) => {
    const userText = (text || inputText).trim()
    if (!userText || loading) return

    const userMsg = { role: 'user', content: userText }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setInputText('')
    setLoading(true)
    setError(null)

    try {
      // Send all messages except the welcome message to the API
      const apiMessages = updatedMessages
        .filter((m) => m !== WELCOME_MESSAGE)
        .slice(-10)
        .map(({ role, content }) => ({ role, content }))

      const data = await sendChatMessage(apiMessages)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.reply },
      ])
    } catch (err) {
      const errMsg =
        err?.response?.data?.detail ||
        'Sorry, HealthBot is unavailable right now. Please try again later.'
      setError(errMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      {/* ── Chat Panel ─────────────────────────────────────────────── */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 w-80 sm:w-96 flex flex-col bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden animate-in slide-in-from-bottom-4 fade-in duration-200">
          {/* Header */}
          <div className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
            <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-bold leading-none">HealthBot</p>
              <p className="text-[11px] text-blue-100 mt-0.5">AI Health Assistant · Powered by Grok</p>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-[10px] text-blue-100">Online</span>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="ml-1 p-1 rounded-lg hover:bg-white/20 transition-colors"
              aria-label="Close chat"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Disclaimer */}
          <div className="flex items-start gap-2 px-3 py-2 bg-amber-50 border-b border-amber-100">
            <AlertCircle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" />
            <p className="text-[10px] text-amber-700 leading-tight">
              General health education only — NOT medical advice. Always consult a healthcare professional.
            </p>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 max-h-72 min-h-[180px] bg-slate-50/50">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Bot className="w-3.5 h-3.5 text-white" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] px-3 py-2 rounded-2xl text-xs leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-sm'
                      : 'bg-white text-slate-700 border border-slate-200 shadow-sm rounded-bl-sm'
                  }`}
                  dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                />
                {msg.role === 'user' && (
                  <div className="w-6 h-6 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <User className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="flex gap-2 justify-start">
                <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-500 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-3.5 h-3.5 text-white" />
                </div>
                <div className="bg-white border border-slate-200 shadow-sm px-3 py-2.5 rounded-2xl rounded-bl-sm flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="flex items-start gap-2 p-2 bg-red-50 border border-red-100 rounded-xl">
                <AlertCircle className="w-3.5 h-3.5 text-red-500 flex-shrink-0 mt-0.5" />
                <p className="text-[11px] text-red-600">{error}</p>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Quick Prompts (only shown if only the welcome message exists) */}
          {messages.length === 1 && (
            <div className="px-3 py-2 border-t border-slate-100 bg-white">
              <p className="text-[10px] font-semibold text-slate-400 mb-1.5 uppercase tracking-wide">Quick questions</p>
              <div className="flex flex-wrap gap-1.5">
                {QUICK_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => handleSend(prompt)}
                    className="text-[11px] px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-100 hover:bg-blue-100 transition-colors font-medium"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="flex items-end gap-2 px-3 py-2.5 border-t border-slate-200 bg-white">
            <textarea
              ref={inputRef}
              rows={1}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask HealthBot anything…"
              className="flex-1 resize-none text-xs rounded-xl border border-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-slate-400 max-h-24 overflow-y-auto"
              style={{ lineHeight: '1.5' }}
              disabled={loading}
            />
            <button
              type="button"
              onClick={() => handleSend()}
              disabled={loading || !inputText.trim()}
              className="flex-shrink-0 w-8 h-8 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white flex items-center justify-center transition-colors"
              aria-label="Send message"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      )}

      {/* ── Floating Action Button ─────────────────────────────────── */}
      <button
        type="button"
        onClick={() => setIsOpen((o) => !o)}
        className={`fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 active:scale-95 ${
          isOpen
            ? 'bg-slate-700 hover:bg-slate-800'
            : 'bg-gradient-to-tr from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700'
        }`}
        aria-label={isOpen ? 'Close HealthBot' : 'Open HealthBot AI Assistant'}
      >
        {isOpen ? (
          <X className="w-6 h-6 text-white" />
        ) : (
          <>
            <MessageCircle className="w-6 h-6 text-white" />
            {/* Sparkle badge */}
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-green-400 rounded-full border-2 border-white flex items-center justify-center">
              <Sparkles className="w-2 h-2 text-white" />
            </span>
          </>
        )}
      </button>
    </>
  )
}
