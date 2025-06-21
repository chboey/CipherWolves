'use client';

import { useEffect, useRef, useState } from 'react';

interface ChatMessage {
  id: string;
  messaging: string;
  botname: string;
  timestamp: string;
}

interface SSEData {
  type: 'status' | 'message' | 'vote_detail' | 'analysis_response' | 'vote';
  data: {
    status?:
      | 'connected'
      | 'keep_alive'
      | 'waiting_for_analysis'
      | 'voting'
      | 'in_progress'
      | 'skipped';
    message: string;
    speaker: string;
    phase: 'communication' | 'voting' | 'analysis';
    agent?: string;
    vote?: string;
    reasoning?: string;
    response?: string;
    action?: string;
    eliminated_agent?: string;
    result?: string | null;
  };
  timestamp: string;
}

export function useSSEChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<
    'disconnected' | 'connecting' | 'connected'
  >('disconnected');
  const eventSourceRef = useRef<EventSource | null>(null);
  const [gameStatus, setGameStatus] = useState<
    'communication' | 'voting' | 'analysis' | 'gameCheck'
  >('communication');

  const connect = () => {
    if (eventSourceRef.current) {
      return; // Already connected
    }

    const apiDomain =
      process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

    const gameID = localStorage.getItem('gameID');

    if (!gameID) {
      console.log('gameID not exists!');
      return;
    }
    setConnectionStatus('connecting');

    try {
      const eventSource = new EventSource(
        `${apiDomain}/games/${gameID}/stream`
      );
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        console.log('SSE connection opened');
      };

      eventSource.onmessage = (event) => {
        try {
          const parsed: SSEData = JSON.parse(event.data);

          // Handle connection status
          if (parsed.type === 'status' && parsed.data.status === 'connected') {
            setIsConnected(true);
            setConnectionStatus('connected');
            console.log('SSE connection success:', parsed.data.message);
            return;
          }

          // Handle chat messages
          if (
            parsed.type === 'message' &&
            parsed.data.speaker &&
            parsed.data.message
          ) {
            const newMessage: ChatMessage = {
              id: crypto.randomUUID(),
              messaging: parsed.data.message,
              botname: parsed.data.speaker,
              timestamp: parsed.timestamp,
            };

            setMessages((prev) => [...prev, newMessage].slice(-100)); // Keep last 100 messages
          }

          if (
            parsed.type === 'vote_detail' &&
            parsed.data.agent &&
            parsed.data.vote &&
            parsed.data.reasoning
          ) {
            const newMessage: ChatMessage = {
              id: crypto.randomUUID(),
              messaging: `I vote ${parsed.data.vote} because ${parsed.data.reasoning}`,
              botname: parsed.data.agent,
              timestamp: parsed.timestamp,
            };

            setMessages((prev) => [...prev, newMessage].slice(-100)); // Keep last 100 messages
          }

          if (
            parsed.type === 'analysis_response' &&
            parsed.data.agent &&
            parsed.data.response
          ) {
            const newMessage: ChatMessage = {
              id: crypto.randomUUID(),
              messaging: `${parsed.data.response}`,
              botname: parsed.data.agent,
              timestamp: parsed.timestamp,
            };

            setMessages((prev) => [...prev, newMessage].slice(-100)); // Keep last 100 messages
          }

          if (parsed.type === 'vote' && parsed.data.action) {
            const newMessage: ChatMessage = {
              id: crypto.randomUUID(),
              messaging:
                parsed.data.eliminated_agent === null
                  ? 'No agent is eliminated'
                  : `Eliminated Agent: ${parsed.data.eliminated_agent}`,
              botname: 'System',
              timestamp: parsed.timestamp,
            };

            setMessages((prev) => [...prev, newMessage].slice(-100)); // Keep last 100 messages
          }

          if (
            parsed.type === 'status' &&
            parsed.data.phase === 'voting' &&
            (parsed.data.status === 'skipped' ||
              parsed.data.status === 'voting')
          ) {
            setGameStatus('voting');
          }
          if (
            parsed.type === 'status' &&
            parsed.data.phase === 'analysis' &&
            parsed.data.status === 'waiting_for_analysis'
          ) {
            setGameStatus('analysis');
          }
          if (
            parsed.type === 'status' &&
            parsed.data.phase === 'communication'
          ) {
            setGameStatus('communication');
          }
        } catch (error) {
          console.error('Error parsing SSE data:', error);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE connection error:', error);
        setIsConnected(false);
        setConnectionStatus('disconnected');

        // Clean up
        eventSource.close();
        eventSourceRef.current = null;
      };
    } catch (error) {
      console.error('Error creating SSE connection:', error);
      setConnectionStatus('disconnected');
    }
  };

  const disconnect = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsConnected(false);
    setConnectionStatus('disconnected');
  };

  const terminateAllConnections = async () => {
    try {
      disconnect();
      setMessages([]); // Clear messages
      console.log('All SSE connections terminated');
    } catch (error) {
      console.error('Error terminating SSE connections:', error);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  return {
    messages,
    isConnected,
    connectionStatus,
    connect,
    disconnect,
    terminateAllConnections,
    gameStatus,
    setGameStatus,
  };
}
