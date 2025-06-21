'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import NextImage from 'next/image';
import { SpriteBot } from '@/components/bot';
import {
  AI_MOVE_DURATION_MAX,
  AI_MOVE_DURATION_MIN,
  AI_MOVE_SPEED,
  AI_PAUSE_DURATION_MAX,
  AI_PAUSE_DURATION_MIN,
  ANIMATION_FRAMES,
  BASE_CANVAS_HEIGHT,
  BASE_CANVAS_WIDTH,
  DIRECTIONS,
  FRAME_DURATION,
  WALKABLE_AREAS,
} from '../../utils/properties';
import { useSSEChat } from '../hooks/use-sse-chat';
import {
  checkGameState,
  startAnalysis,
  StartGameRound,
  startVoting,
} from '@/actions/game-phase';
import beerMug from '@images/beer-mug.svg';
import cipher from '@images/cipher-logo2.svg';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useRouter } from 'next/navigation';
import { ScrollArea } from '@radix-ui/react-scroll-area';

interface BotState {
  x: number;
  y: number;
  direction: 'up' | 'down' | 'left' | 'right';
  isMoving: boolean;
  message: string | null;
  messageTimeoutId?: NodeJS.Timeout;
  moveDirection?: { x: number; y: number };
  ai: {
    action: 'moving' | 'paused';
    actionEndTime: number;
  };
}

interface GameData {
  game_id: string;
  status: string;
  current_round: number;
  current_phase: string;
  remaining_agents: string[];
  eliminated_agents: string[];
  werewolf: string;
  result: string | null;
}

interface CanvasDimensions {
  width: number;
  height: number;
  scaleFactor: number;
}

// Spawn points in walkable areas
const SPAWN_POINTS = [
  { x: 330, y: 250 }, // top left
  { x: 800, y: 370 }, // left center pathway
  { x: 100, y: 390 }, // right center pathway
  { x: 500, y: 500 }, // center
  { x: 700, y: 500 }, // center
];

function getRandomInt(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Enhanced position validation
function isValidPosition(
  x: number,
  y: number,
  botStates: BotState[],
  excludeIndex = -1
): boolean {
  // Check canvas boundaries
  const margin = 30;
  if (
    x < margin ||
    x > BASE_CANVAS_WIDTH - margin ||
    y < margin ||
    y > BASE_CANVAS_HEIGHT - margin
  ) {
    return false;
  }

  // Check if in walkable area
  if (!SpriteBot.isInWalkableArea(x, y)) {
    return false;
  }

  // Check collision with other bots
  if (SpriteBot.checkBotCollision(x, y, botStates, excludeIndex)) {
    return false;
  }

  return true;
}

function getSpriteIconPath(name: string, index: number): string {
  const spriteMapping: { [key: string]: string } = {
    agent_Alice: '/alice_icon.svg',
    agent_Bob: '/bob_icon.svg',
    agent_Charlie: '/charlie_icon.svg',
    agent_Dom: '/dom_icon.svg',
    agent_Elise: '/elise_icon.svg',
  };

  // Return specific sprite for the character, or fallback to index-based naming
  return spriteMapping[name] || `/sprites/character_${index + 1}.png`;
}

function getRandomDirection(): { x: number; y: number } {
  const directions = [
    { x: 0, y: -1 }, // up
    { x: -1, y: 0 }, // left
    { x: 0, y: 1 }, // down
    { x: 1, y: 0 }, // right
  ];
  return directions[Math.floor(Math.random() * directions.length)];
}

const GamePage = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const botsRef = useRef<SpriteBot[] | null>(null);
  const botStatesRef = useRef<BotState[]>([]);
  const animationIdRef = useRef<number | null>(null);
  const backgroundImageRef = useRef<HTMLImageElement | null>(null);
  const initializationRef = useRef<boolean>(false);

  const [isInitialized, setIsInitialized] = useState(false);
  const [botStates, setBotStates] = useState<BotState[]>([]);
  const [animationFrame, setAnimationFrame] = useState<number>(0);
  const [backgroundLoaded, setBackgroundLoaded] = useState(false);
  const [canvasDimensions, setCanvasDimensions] = useState<CanvasDimensions>({
    width: BASE_CANVAS_WIDTH,
    height: BASE_CANVAS_HEIGHT,
    scaleFactor: 1,
  });
  const [userInput, setUserInput] = useState<string>('');
  const [finalResult, setFinalResult] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameData | null>(null);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [gameRound, setGameRound] = useState(1);
  const router = useRouter();

  const {
    messages: chatMessages,
    isConnected: sseConnected,
    connectionStatus,
    connect: connectSSE,
    disconnect: disconnectSSE,
    terminateAllConnections,
    gameStatus,
    setGameStatus,
  } = useSSEChat();

  // Load background image
  useEffect(() => {
    const loadBackground = async () => {
      try {
        const img = new Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
          backgroundImageRef.current = img;
          setBackgroundLoaded(true);
          console.log('Background image loaded successfully');
        };

        img.onerror = (error) => {
          console.error('Failed to load background image:', error);
          setBackgroundLoaded(true); // Continue even if background fails
        };

        img.src = '/map.png';
      } catch (error) {
        console.error('Error loading background image:', error);
        setBackgroundLoaded(true); // Continue even if background fails
      }
    };

    loadBackground();
  }, []);

  // SSE Connection Management
  useEffect(() => {
    if (
      backgroundLoaded &&
      !sseConnected &&
      connectionStatus === 'disconnected'
    ) {
      connectSSE();
    }
  }, [
    backgroundLoaded,
    sseConnected,
    connectionStatus,
    connectSSE,
    disconnectSSE,
  ]);

  const updateCanvasSize = useCallback(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const containerWidth = container.clientWidth - 32;
    const containerHeight = window.innerHeight - 200;

    const aspectRatio = BASE_CANVAS_WIDTH / BASE_CANVAS_HEIGHT;
    let newWidth = Math.min(containerWidth, 1000);
    let newHeight = newWidth / aspectRatio;

    if (newHeight > containerHeight) {
      newHeight = containerHeight;
      newWidth = containerHeight * aspectRatio;
    }

    const scaleFactor = Math.min(
      newWidth / BASE_CANVAS_WIDTH,
      newHeight / BASE_CANVAS_HEIGHT
    );

    setCanvasDimensions({
      width: newWidth,
      height: newHeight,
      scaleFactor,
    });
  }, []);

  useEffect(() => {
    updateCanvasSize();
    const handleResize = () => updateCanvasSize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [updateCanvasSize]);

  const drawBackground = useCallback(
    (ctx: CanvasRenderingContext2D, scaleFactor: number) => {
      if (backgroundLoaded && backgroundImageRef.current) {
        // Draw the background image
        ctx.drawImage(
          backgroundImageRef.current,
          0,
          0,
          BASE_CANVAS_WIDTH * scaleFactor,
          BASE_CANVAS_HEIGHT * scaleFactor
        );

        // Draw walkable area overlays (semi-transparent for debugging)
        if (process.env.NODE_ENV === 'development') {
          ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
          WALKABLE_AREAS.forEach((area) => {
            ctx.fillRect(
              area.x * scaleFactor,
              area.y * scaleFactor,
              area.width * scaleFactor,
              area.height * scaleFactor
            );
          });
        }
      } else {
        // Fallback background
        ctx.fillStyle = '#8B4513';
        ctx.fillRect(
          0,
          0,
          BASE_CANVAS_WIDTH * scaleFactor,
          BASE_CANVAS_HEIGHT * scaleFactor
        );

        // Draw walkable areas as green rectangles
        ctx.fillStyle = '#228B22';
        WALKABLE_AREAS.forEach((area) => {
          ctx.fillRect(
            area.x * scaleFactor,
            area.y * scaleFactor,
            area.width * scaleFactor,
            area.height * scaleFactor
          );
        });
      }
    },
    [backgroundLoaded]
  );

  const drawGame = useCallback(() => {
    if (
      !isInitialized ||
      !canvasRef.current ||
      !botsRef.current ||
      botStatesRef.current.length === 0
    ) {
      return;
    }

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const { scaleFactor } = canvasDimensions;

    ctx.clearRect(0, 0, canvasDimensions.width, canvasDimensions.height);

    // Draw background
    drawBackground(ctx, scaleFactor);

    // Draw each bot
    botsRef.current.forEach((bot, index) => {
      const botState = botStatesRef.current[index];
      if (!botState) return;

      const isEliminated =
        gameState?.eliminated_agents?.includes(bot.name) || false;
      if (isEliminated) return; // Skip drawing eliminated bots

      bot.draw(
        ctx,
        botState.x,
        botState.y,
        botState.direction,
        animationFrame,
        botState.isMoving,
        scaleFactor,
        botState.message
      );
    });
  }, [
    isInitialized,
    animationFrame,
    canvasDimensions,
    drawBackground,
    gameState?.eliminated_agents,
  ]);

  // Initialize bots - Only run once
  useEffect(() => {
    if (initializationRef.current || !backgroundLoaded) return;

    const initializeGame = () => {
      console.log('Initializing game...');
      initializationRef.current = true;

      // Clean up any existing bots
      if (botsRef.current) {
        botsRef.current.forEach((bot) => bot.destroy?.());
      }

      const botConfigs = [
        { name: 'agent_Alice', sprite: '/sprites/alice.png' },
        { name: 'agent_Bob', sprite: '/sprites/bob.png' },
        { name: 'agent_Charlie', sprite: '/sprites/charlie.png' },
        { name: 'agent_Dom', sprite: '/sprites/dom.png' },
        { name: 'agent_Elise', sprite: '/sprites/elise.png' },
      ];

      botsRef.current = botConfigs.map(
        (config, index) =>
          new SpriteBot(index, config.name, () => {}, config.sprite) // Empty callback since we handle messages via SSE
      );

      const initialBotStates: BotState[] = [];

      botsRef.current.forEach((_, index) => {
        const spawnPoint = SPAWN_POINTS[index % SPAWN_POINTS.length];
        const offsetX = (Math.random() - 0.5) * 40;
        const offsetY = (Math.random() - 0.5) * 40;

        initialBotStates.push({
          x: spawnPoint.x + offsetX,
          y: spawnPoint.y + offsetY,
          direction: DIRECTIONS[index % DIRECTIONS.length],
          isMoving: false,
          message: null,
          ai: {
            action: 'paused',
            actionEndTime:
              Date.now() +
              getRandomInt(AI_PAUSE_DURATION_MIN, AI_PAUSE_DURATION_MAX),
          },
        });
      });

      botStatesRef.current = initialBotStates;
      setBotStates(initialBotStates);
      setIsInitialized(true);
      console.log('Game initialized successfully!');
    };

    initializeGame();

    return () => {
      if (botsRef.current) {
        botsRef.current.forEach((bot) => bot.destroy?.());
        botsRef.current = null;
      }
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = null;
      }
      initializationRef.current = false;
    };
  }, [backgroundLoaded]);

  // Animation loop
  useEffect(() => {
    if (!isInitialized) return;

    let lastUpdateTime = 0;
    let lastFrameTime = 0;

    const animate = (timestamp: number) => {
      if (lastUpdateTime === 0) {
        lastUpdateTime = timestamp;
        lastFrameTime = timestamp;
      }

      const deltaTime = timestamp - lastUpdateTime;
      const frameDelta = timestamp - lastFrameTime;

      if (frameDelta >= FRAME_DURATION) {
        setAnimationFrame((prev) => (prev + 1) % (ANIMATION_FRAMES * 15));
        lastFrameTime = timestamp;
      }

      const currentBotStates = [...botStatesRef.current];
      let stateChanged = false;

      currentBotStates.forEach((botState, index) => {
        const now = Date.now();

        if (now >= botState.ai.actionEndTime) {
          if (botState.ai.action === 'paused') {
            botState.moveDirection = getRandomDirection();
            botState.ai.action = 'moving';
            botState.ai.actionEndTime =
              now + getRandomInt(AI_MOVE_DURATION_MIN, AI_MOVE_DURATION_MAX);
            botState.isMoving = true;

            if (
              Math.abs(botState.moveDirection.x) >
              Math.abs(botState.moveDirection.y)
            ) {
              botState.direction =
                botState.moveDirection.x > 0 ? 'right' : 'left';
            } else {
              botState.direction = botState.moveDirection.y > 0 ? 'down' : 'up';
            }

            stateChanged = true;
          } else {
            botState.ai.action = 'paused';
            botState.ai.actionEndTime =
              now + getRandomInt(AI_PAUSE_DURATION_MIN, AI_PAUSE_DURATION_MAX);
            botState.isMoving = false;
            botState.moveDirection = undefined;
            stateChanged = true;
          }
        }

        if (botState.ai.action === 'moving' && botState.moveDirection) {
          const moveSpeed = AI_MOVE_SPEED * (deltaTime / 16.67);
          const newX = botState.x + botState.moveDirection.x * moveSpeed;
          const newY = botState.y + botState.moveDirection.y * moveSpeed;

          if (isValidPosition(newX, newY, currentBotStates, index)) {
            botState.x = newX;
            botState.y = newY;
            stateChanged = true;
          } else {
            // Try to find a new valid direction
            let attempts = 0;
            let foundValidDirection = false;

            while (attempts < 8 && !foundValidDirection) {
              const newDirection = getRandomDirection();
              const testX = botState.x + newDirection.x * moveSpeed;
              const testY = botState.y + newDirection.y * moveSpeed;

              if (isValidPosition(testX, testY, currentBotStates, index)) {
                botState.moveDirection = newDirection;

                if (Math.abs(newDirection.x) > Math.abs(newDirection.y)) {
                  botState.direction = newDirection.x > 0 ? 'right' : 'left';
                } else {
                  botState.direction = newDirection.y > 0 ? 'down' : 'up';
                }

                foundValidDirection = true;
              }
              attempts++;
            }

            if (!foundValidDirection) {
              botState.ai.action = 'paused';
              botState.ai.actionEndTime = now + getRandomInt(1000, 2000);
              botState.isMoving = false;
              botState.moveDirection = undefined;
            }

            stateChanged = true;
          }
        }
      });

      if (stateChanged) {
        botStatesRef.current = currentBotStates;
        setBotStates([...currentBotStates]);
      }

      lastUpdateTime = timestamp;
      drawGame();
      animationIdRef.current = requestAnimationFrame(animate);
    };

    animationIdRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = null;
      }
    };
  }, [isInitialized, drawGame]);

  // Handle SSE messages and show speech bubbles on corresponding sprites
  useEffect(() => {
    if (chatMessages.length === 0 || !botsRef.current) return;

    // Get the latest message
    const latestMessage = chatMessages[chatMessages.length - 1];

    // Find the bot index by matching the name
    const botIndex = botsRef.current.findIndex(
      (bot) =>
        bot.name === latestMessage.botname ||
        bot.name.includes(latestMessage.botname.replace('agent_', ''))
    );

    if (botIndex !== -1) {
      // Update the bot state to show "..." message bubble
      setBotStates((prevStates) => {
        const newStates = [...prevStates];
        const currentState = newStates[botIndex];

        // Clear any existing timeout
        if (currentState?.messageTimeoutId) {
          clearTimeout(currentState.messageTimeoutId);
        }

        // Set the message to "..." and create timeout to clear it
        const timeoutId = setTimeout(() => {
          setBotStates((states) => {
            const clearedStates = [...states];
            if (clearedStates[botIndex]) {
              clearedStates[botIndex] = {
                ...clearedStates[botIndex],
                message: null,
                messageTimeoutId: undefined,
              };
              botStatesRef.current = clearedStates;
            }
            return clearedStates;
          });
        }, 6000);

        newStates[botIndex] = {
          ...currentState,
          message: '...',
          messageTimeoutId: timeoutId,
        };

        botStatesRef.current = newStates;
        return newStates;
      });
    }
  }, [chatMessages]);

  const handleAnalysisSubmit = async (userInput: string) => {
    setSubmitting(true);
    const gameID = localStorage.getItem('gameID');
    if (!gameID) return;

    try {
      await startAnalysis(gameID, userInput);
      setUserInput('');
      setGameStatus('voting');
      setSubmitting(false);
    } catch (error) {
      console.log('Analysis error:', error);
    }
  };

  useEffect(() => {
    const updateGameState = async () => {
      const gameID = localStorage.getItem('gameID');
      if (!gameID) return;

      try {
        const data = await checkGameState(gameID);
        setGameState(data);
        if (data.result !== null) {
          setFinalResult(data.result);
          terminateAllConnections();
          router.push('/post-game');
        }
      } catch (error) {
        console.error('Error updating game state:', error);
      }
    };

    // Update game state when game status changes
    if (sseConnected) {
      updateGameState();
    }
  }, [sseConnected, gameStatus]);

  useEffect(() => {
    if (!sseConnected && connectionStatus !== 'connected') {
      return;
    }

    const gameID = localStorage.getItem('gameID');
    if (!gameID) {
      console.log('gameID does not exist!');
      return;
    }

    const handleGameFlow = async () => {
      try {
        switch (gameStatus) {
          case 'communication':
            await StartGameRound(gameID);
            break;

          case 'voting':
            await startVoting(gameID);
            setGameRound((prev) => prev + 1);
            break;

          // Don't handle 'analysis' here - it needs user input
          default:
            break;
        }
      } catch (error) {
        console.log('Game flow error:', error);
      }
    };

    // Only execute automated phases if we don't have a result yet
    if (finalResult === null && gameStatus !== 'analysis') {
      handleGameFlow();
    }
  }, [gameStatus, finalResult, sseConnected, connectionStatus]);

  return (
    <div className="min-h-screen bg-[#15130A] p-2 md:p-4">
      <div className="flex flex-col lg:flex-row gap-4 max-w-7xl mx-auto">
        <div
          ref={containerRef}
          className="flex-1 shadow-[0_2px_8px_rgb(0,0,0,0.2)] shadow-[#FF990050] border-0 bg-[#1a1810] text-white rounded-xl p-6 min-h-0"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-2">
            <div className="flex items-center gap-4">
              <h1 className="flex flex-row text-2xl md:text-3xl font-bold gap-2 items-center">
                <NextImage src={cipher} alt="icon" className="w-10 h-10" />
                CipherWolves
              </h1>
            </div>
            <div className="text-xs font-medium text-black bg-[#ffb300] w-fit px-3 py-1 rounded-full">
              Scale: {canvasDimensions.scaleFactor.toFixed(2)}x
            </div>
          </div>

          {!isInitialized ? (
            <div className="flex items-center justify-center h-64 md:h-96">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-amber-600 border-t-transparent mx-auto mb-4"></div>
                <div className="text-lg font-medium text-gray-700">
                  Loading medieval town...
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  {backgroundLoaded
                    ? 'Setting up characters...'
                    : 'Loading background...'}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex justify-center">
              <canvas
                ref={canvasRef}
                width={canvasDimensions.width}
                height={canvasDimensions.height}
                className=" rounded-lg shadow-lg"
                style={{
                  width: canvasDimensions.width,
                  height: canvasDimensions.height,
                  imageRendering: 'pixelated',
                }}
              />
            </div>
          )}

          {/* Bot Info Grid */}
          <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            {botsRef.current?.slice(0, 6).map((bot, index) => {
              const isEliminated =
                gameState?.eliminated_agents?.includes(bot.name) || false;

              const latestMessageBotName =
                chatMessages.length > 0
                  ? chatMessages[chatMessages.length - 1].botname.split(':')[0]
                  : null;

              // Check if this bot is the one who sent the latest message
              const isLatestSpeaker = latestMessageBotName === bot.name;

              const characterImagePath = getSpriteIconPath(bot.name, index);
              return (
                <div
                  key={bot.name}
                  className={`p-3 rounded-lg border ${
                    isEliminated
                      ? 'bg-gray-100 opacity-60'
                      : 'bg-gradient-to-r from-amber-50 to-orange-50'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div
                      className={`w-10 h-10 rounded-full border-2 flex items-center justify-center overflow-hidden ${
                        isEliminated ? 'border-gray-400' : 'border-gray-300'
                      }`}
                      style={{
                        borderColor: isEliminated ? '#999' : bot.color,
                      }}
                    >
                      {isEliminated ? (
                        <div className="w-10 h-10 bg-gray-300 rounded-full"></div>
                      ) : (
                        <NextImage
                          src={characterImagePath}
                          alt={`${bot.name} icon`}
                          width={6}
                          height={6}
                          className="w-10 h-10 object-cover rounded-full"
                        />
                      )}
                    </div>
                    <div
                      className={`font-semibold ${
                        isEliminated
                          ? 'text-gray-500 line-through'
                          : 'text-gray-800'
                      }`}
                    >
                      {bot.name}
                    </div>
                    {isEliminated && (
                      <span className="text-red-500 text-xs">💀</span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {isEliminated
                      ? '💀 Eliminated'
                      : botStates[index]?.isMoving
                      ? '🚶 Moving'
                      : '⏸️ Paused'}
                  </div>

                  {!isLatestSpeaker &&
                    chatMessages.length > 0 &&
                    !isEliminated && (
                      <div className="text-gray-600 text-xs mt-1">
                        🔄 Suspicion & Trust level updating...
                      </div>
                    )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Chat Panel */}
        <div className="w-full lg:w-124 shadow-[0_2px_8px_rgb(0,0,0,0.2)] shadow-[#FF990050] border-0 bg-[#1a1810] rounded-xl p-6 flex flex-col">
          <div className="flex items-center gap-2 mb-4">
            <h3 className="text-xl font-bold text-white flex flex-row items-center space-y-3">
              <NextImage src={beerMug} alt="townchat" className="w-8 h-8" />{' '}
              Town Chat
            </h3>
            <div
              className={`text-xs px-2 py-1 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-green-100 text-green-800'
                  : connectionStatus === 'connecting'
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
              }`}
            >
              {connectionStatus === 'connected'
                ? '🟢 Connected'
                : connectionStatus === 'connecting'
                ? '🟡 Connecting...'
                : '🔴 Disconnected'}
            </div>
            <div className="bg-amber-100 text-amber-800 text-xs px-2 py-1 rounded-full">
              {chatMessages.length} messages
            </div>
          </div>

          <div className="flex-1 min-h-0 mb-3">
            <div className="h-80 lg:h-116 overflow-y-auto border border-amber-500/20 rounded-lg p-3 space-y-3 bg-[#2D2B22FF]">
              {chatMessages.length === 0 ? (
                <div className="text-white/60 text-center py-12">
                  <div className="text-4xl mb-2">🏰</div>
                  <div className="font-medium">
                    Waiting for town conversations...
                  </div>
                  <div className="text-sm mt-1">
                    Medieval characters will start chatting soon!
                  </div>
                </div>
              ) : (
                chatMessages.map((msg) => {
                  const bot = botsRef.current?.find(
                    (b) => b.name === msg.botname.split(':')[0]
                  );
                  return (
                    <ScrollArea key={msg.id}>
                      <div
                        key={msg.id}
                        className="bg-[#2a2520] p-3 rounded-lg shadow-sm border-l-4"
                        style={{ borderLeftColor: bot?.color || '#ccc' }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: bot?.color || '#ccc' }}
                            ></div>
                            <div className="text-sm font-semibold text-white">
                              {msg.botname.split(':')[0]}
                            </div>
                          </div>
                          <div className="text-xs text-white">
                            {new Date(msg.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                        <div className="text-sm text-white leading-relaxed">
                          {msg.messaging}
                        </div>
                      </div>
                    </ScrollArea>
                  );
                })
              )}
            </div>
          </div>

          <div className="flex gap-4">
            <Input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              className="border-2 text-white"
              placeholder="Enter your analysis..."
              disabled={gameStatus !== 'analysis'}
            />
            <Button
              variant="outline"
              onClick={() => handleAnalysisSubmit(userInput)}
              disabled={gameStatus !== 'analysis' || submitting}
              className="flex items-center gap-2"
            >
              Send
            </Button>
          </div>

          <div className="mt-4 space-y-2">
            <div className="grid grid-cols-2 gap-4 text-xs text-gray-500">
              <div className="bg-[#2a2520] p-2 rounded">
                <div className="font-medium text-white/70">Game Phase</div>
                <div className="text-base font-bold text-amber-500">
                  {gameStatus}
                </div>
              </div>
              <div className="bg-[#2a2520] p-2 rounded">
                <div className="font-medium text-white/70">Round</div>
                <div className="text-xl font-bold text-amber-500">
                  {gameRound}
                </div>
              </div>
              <div className="bg-[#2a2520] p-2 rounded">
                <div className="font-medium text-white/70">Alive Agents</div>
                <div className="text-xl font-bold text-amber-500">
                  {gameState?.remaining_agents?.length || 0}
                </div>
              </div>
              <div className="bg-[#2a2520] p-2 rounded">
                <div className="font-medium text-white/70">
                  Eliminated Agents
                </div>
                <div className="text-xl font-bold text-amber-500">
                  {gameState?.eliminated_agents?.length || 0}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GamePage;
