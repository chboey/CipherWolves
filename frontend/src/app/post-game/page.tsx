'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Home,
  RotateCcw,
  MessageSquare,
  Clock,
  Target,
  Hash,
  User,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import PostGameSkeleton from '@/components/skeletonLoading';
import lightbulb from '@images/lightbulb.svg';
import instruction from '@images/instruction.svg';
import {
  analyzeConversation,
  getAgentPersonas,
  getConversationHistory,
  getKeywords,
  getDataMetrics,
} from '@/actions/post-game-analysis';
import Image from 'next/image';
import cipher from '@images/cipher-logo2.svg';
import helmet from '@images/helmet.svg';
import werewolf from '@images/werewolf.svg';
import dead from '@images/dead.svg';
import love from '@images/love.svg';
import { checkGameState } from '@/actions/game-phase';
import { createGame } from '@/actions/create-game';
import GameChart, { AgentMetrics } from '@/components/metricsChart';
import flag from '@images/flag.svg';
import { Download } from 'lucide-react';
import { useDataDownload } from '../hooks/use-download';

interface Message {
  speaker: string;
  message: string;
  type?: 'voting_result' | 'vote' | 'analysis';
  timestamp: string;
}

export interface Conversation {
  round: number;
  phase: string;
  messages: Message[];
}

interface Persona {
  agent_name: string;
  instruction: string;
  description: string;
  is_werewolf: boolean;
}

interface GameState {
  game_id: string;
  status: string;
  current_round: number;
  current_phase: string;
  remaining_agents: string[];
  eliminated_agents: string[];
  werewolf: string;
  result: string;
}

interface Summary {
  agent_name: string;
  behavior_analysis: string;
  key_actions: string[];
  suspicious_patterns: string[];
  confidence_score: number;
}

export interface GameData {
  conversation: Conversation[];
  personas: Persona[];
  keywords: Array<string>;
  gameStates: GameState;
  metrics: AgentMetrics[];
  agent_analyses: Summary[];
}
export default function PostGamePage() {
  const [selectedCharacter, setSelectedCharacter] = useState(0);
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<GameData | null>(null);

  const characterIconMap: { [key: string]: any } = {
    agent_Alice: 'alice_icon.svg',
    agent_Bob: 'bob_icon.svg',
    agent_Charlie: 'charlie_icon.svg',
    agent_Dom: 'dom_icon.svg',
    agent_Elise: 'elise_icon.svg',
  };

  const { downloadData } = useDataDownload();

  const handleDownload = () => {
    downloadData(
      data?.keywords,
      data?.conversation,
      data?.metrics,
      'cipherwolves-data'
    );
  };

  const getCharacterIcon = (persona: Persona | undefined) => {
    if (!persona) {
      return;
    }
    return characterIconMap[persona?.agent_name] || '/placeholder.svg';
  };

  const FilteredSelectedCharacterMessages = () => {
    if (!data?.conversation || !data?.personas[selectedCharacter]) return null;

    const selectedCharacterName = data.personas[selectedCharacter].agent_name;

    // Filter messages that involve the selected character (either as speaker or mentioned)
    const characterMessages = data.conversation.flatMap((conv) =>
      getFilteredMessages(conv.messages)
        .filter(
          (msg) =>
            msg.speaker === selectedCharacterName ||
            msg.message
              .toLowerCase()
              .includes(selectedCharacterName.toLowerCase())
        )
        .map((msg) => ({
          ...msg,
          round: conv.round,
          phase: conv.phase,
        }))
    );

    if (characterMessages.length === 0) {
      return (
        <div className="text-center text-white/70 py-8">
          <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p>No conversations found for {selectedCharacterName}</p>
        </div>
      );
    }

    return (
      <div className="space-y-3">
        {characterMessages.map((entry, index) => (
          <div key={index} className="flex flex-col gap-1">
            <div className="flex items-center gap-2 p-3 text-sm text-white/70 rounded-[6px] bg-[#2a2520]">
              <Clock className="h-3 w-3" />
              Round {entry.round} : {entry.phase} - {entry.speaker}
              {entry.speaker === selectedCharacterName && (
                <Badge
                  variant="secondary"
                  className="text-xs bg-amber-500/20 text-amber-400 border-amber-500/30"
                >
                  Speaking
                </Badge>
              )}
              {entry.speaker !== selectedCharacterName && (
                <Badge
                  variant="secondary"
                  className="text-xs bg-blue-500/20 text-blue-400 border-blue-500/30"
                >
                  Mentioned
                </Badge>
              )}
            </div>
            <p className="text-sm font-medium text-white p-3 rounded-[6px] bg-[#2a2520]">
              {entry.message}
            </p>
            {index < characterMessages.length - 1 && (
              <Separator className="my-2 bg-white/20" />
            )}
          </div>
        ))}
      </div>
    );
  };

  const getFilteredMessages = (messages: Message[]) => {
    return messages.filter(
      (msg) =>
        !msg.type || // No type (communication phase)
        msg.type === 'voting_result' ||
        msg.type === 'vote' ||
        msg.type === 'analysis'
    );
  };

  const FilteredMessages = () => {
    if (!data?.conversation) return null;

    const Messages = data.conversation.flatMap((conv) =>
      getFilteredMessages(conv.messages).map((msg) => ({
        ...msg,
        round: conv.round,
        phase: conv.phase,
      }))
    );

    return (
      <div className="space-y-3">
        {Messages.map((entry, index) => (
          <div key={index} className="flex flex-col gap-1">
            <div className="flex items-center gap-2 p-2 text-sm text-white/70 rounded-[6px] bg-[#2a2520]">
              <Clock className="h-3 w-3" />
              {new Date(entry.timestamp).toLocaleTimeString()}
              {' - '}
              Round {entry.round} : {entry.phase} - {entry.speaker}
            </div>
            <p className="text-sm font-medium text-white p-2 rounded-[6px] bg-[#2a2520]">
              {entry.message}
            </p>
            {index < entry.message.length - 1 && (
              <Separator className="my-2 bg-white/20" />
            )}
          </div>
        ))}
      </div>
    );
  };

  const getPersonaRole = (persona: Persona): string => {
    const words = persona.instruction.split(' ');
    const fourthWord = words[3] || '';
    return fourthWord.replace(/[.,;:!?]$/, '');
  };

  const handleRedirect = async (direct: string) => {
    if (direct === 'game') {
      try {
        // Check if API response indicates success
        const data = await createGame();
        if (data.game_id) {
          console.log('Game started successfully:', data.game_id);
          localStorage.setItem('gameID', data.game_id);
          // Route to game page on success
          router.push('/game');
        } else {
          console.error('Game start failed:', data.message);
          // Handle API error response
          alert('Failed to start game. Please try again.');
        }
      } catch (error) {
        console.error('Error starting game:', error);
        alert('Network error. Please check your connection and try again.');
      }
    } else {
      router.push('/');
    }
  };

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true);

        const gameID = localStorage.getItem('gameID');
        if (!gameID) {
          console.log('gameID not exist!');
          return;
        }

        // Wait for all API calls to complete
        const [
          conversation,
          personas,
          keywords,
          gameStates,
          metrics,
          agent_analyses,
        ] = await Promise.all([
          getConversationHistory(gameID),
          getAgentPersonas(gameID),
          getKeywords(gameID),
          checkGameState(gameID),
          getDataMetrics(gameID),
          analyzeConversation(gameID),
        ]);

        // Set all data at once
        setData({
          conversation,
          personas,
          keywords,
          gameStates,
          metrics,
          agent_analyses,
        });

        console.log(data);
      } catch (error) {
        console.log('error fetching data from APIs', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAllData();
  }, []);

  if (loading) {
    return <PostGameSkeleton />;
  }

  return (
    <div className="min-h-screen bg-[#15130A]">
      {/* Header */}
      <header className="bg-[#15130A] shadow-xl shadow-[#FF990050]">
        <div className="container mx-auto md:px-4 py-5">
          <div className="flex flex-col sm:flex-row items-center sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4 text-white">
              <Image src={cipher} alt="logo" className="w-10 h-10" />
              <h1 className="text-2xl font-bold md:display hidden">
                CipherWolves
              </h1>
              <h1 className="text-xl font-bold">Post-Game Analysis</h1>
              <Badge
                variant="secondary"
                className="flex items-center gap-1 bg-[#ffb300]"
              >
                <Target className="h-3 w-3" />
                Keywords: 300
              </Badge>
            </div>
            <div className="flex gap-4">
              <Button
                variant="outline"
                className="bg-[#ffb300] border-0 flex items-center gap-2 hover:scale-103 transition-all duration-300 transform hover:bg-[#ffb300]"
                onClick={handleDownload}
              >
                <span className="hidden md:inline">Download</span>
                <Download className="h-4 w-4 md:hidden" />
              </Button>
              <Button
                variant="outline"
                className="flex items-center gap-2 shadow-[4.0px_4.0px_0.0px_rgba(0,0,0,0.38)] hover:scale-103 transition-all duration-300 transform hover:shadow-[#ffb300]"
                onClick={() => handleRedirect('game')}
              >
                <RotateCcw className="h-4 w-4" />
                New Game
              </Button>
              <Button
                variant="outline"
                className="flex items-center gap-2 shadow-[4.0px_4.0px_0.0px_rgba(0,0,0,0.38)] hover:scale-103 transition-all duration-300 transform hover:shadow-[#ffb300]"
                onClick={() => handleRedirect('home')}
              >
                <Home className="h-4 w-4" />
                Return to MainPage
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left Sidebar - Full Conversation Log & Keywords */}
          <div className="lg:col-span-2 space-y-6">
            {/* Keywords Section */}
            <Card className="shadow-[0_2px_8px_rgb(0,0,0,0.2)] shadow-[#FF990050] border-0 bg-[#1a1810] text-white">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2 tex-white">
                  <Hash className="h-5 w-5" />
                  Game Keywords
                </CardTitle>
                <CardDescription className="text-white/70">
                  All 300 keywords from the game session
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-80">
                  <div className="flex flex-wrap gap-1">
                    {data?.keywords.map((keyword, index) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className="text-xs shadow-sm border-white/20 hover:bg-white/10 text-white transition-all border-2 hover:shadow-amber-500/60 hover:border-[#ffb300]"
                      >
                        {keyword}
                      </Badge>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            {/* Full Conversation Log */}
            <Card className="shadow-[0_2px_8px_rgb(0,0,0,0.2)] shadow-[#FF990050] border-0 bg-[#1a1810]">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <MessageSquare className="h-5 w-5" />
                  Full Game Log
                </CardTitle>
                <CardDescription className="text-white/70">
                  Complete conversation history
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[600px] text-black bg-[#1a1810]">
                  {FilteredMessages()}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* Main Content - Character Analysis */}
          <div className="lg:col-span-3 text-white">
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <User className="w-5 h-5" />
                  Character Analysis
                </h2>

                {/* Character Selection */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-6">
                  {data?.personas.map((persona, index) => (
                    <Card
                      key={persona.agent_name}
                      className={`cursor-pointer transition-all shadow-md border-2 hover:shadow-amber-400/60 hover:border-[#ffb300] bg-[#1a1810]${
                        selectedCharacter === index
                          ? 'border-2 border-amber-500/80'
                          : ''
                      }`}
                      onClick={() => setSelectedCharacter(index)}
                    >
                      <CardContent className="p-3">
                        <div className="flex flex-col items-center text-center gap-2">
                          <Avatar className="h-14 w-14">
                            <AvatarImage
                              src={getCharacterIcon(persona)}
                              alt={persona.agent_name}
                            />
                            <AvatarFallback>
                              {persona.agent_name
                                .split('')
                                .map((n) => n[0])
                                .join('')}
                            </AvatarFallback>
                          </Avatar>
                          <div className="text-white">
                            <p className="font-medium text-sm">
                              {persona.agent_name.split('_')[1]}
                            </p>
                            <p className="text-xs text-white/60">
                              {getPersonaRole(persona)}
                            </p>
                          </div>
                          <div className="flex items-center gap-1 text-xs text-white">
                            {persona.is_werewolf ? (
                              <Image
                                src={werewolf}
                                alt="werewolf"
                                className="w-4 h-4"
                              />
                            ) : (
                              <Image
                                src={helmet}
                                alt="villager"
                                className="w-4 h-4 text-white"
                              />
                            )}
                            {persona.is_werewolf ? 'WereWolf' : 'Villager'}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>

              {/* Selected Character Details */}

              <Card className="shadow-[0_2px_8px_rgb(0,0,0,0.2)] shadow-[#FF990050] border-0 bg-[#1a1810]">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <Avatar className="h-16 w-16">
                      <AvatarImage
                        src={getCharacterIcon(
                          data?.personas[selectedCharacter]
                        )}
                        alt={data?.personas[selectedCharacter]?.agent_name}
                      />
                      <AvatarFallback>
                        {data?.personas[selectedCharacter]?.agent_name
                          .split(' ')
                          .map((n) => n[0])
                          .join('')}
                      </AvatarFallback>
                    </Avatar>
                    <div className="space-y-3">
                      <CardTitle className="text-white font-bold text-lg">
                        {
                          data?.personas[selectedCharacter]?.agent_name.split(
                            '_'
                          )[1]
                        }
                      </CardTitle>
                      <CardDescription className="text-white font-medium">
                        {data?.personas[selectedCharacter] &&
                          getPersonaRole(data.personas[selectedCharacter])}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="personas" className="w-full">
                    <TabsList className="grid w-full grid-cols-3 bg-white/10 ">
                      <TabsTrigger
                        value="personas"
                        className="text-white data-[state=active]:bg-[#ffb300] data-[state=active]:text-black"
                      >
                        Personas & Analysis
                      </TabsTrigger>
                      <TabsTrigger
                        value="conversations"
                        className="text-white data-[state=active]:bg-[#ffb300] data-[state=active]:text-black"
                      >
                        Conversations
                      </TabsTrigger>
                      <TabsTrigger
                        value="chart"
                        className="text-white data-[state=active]:bg-[#ffb300] data-[state=active]:text-black"
                      >
                        Metrics
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="personas" className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                        <div className="text-center p-3 bg-[#2a2520] rounded-lg flex flex-col items-center justify-center">
                          <div className="text-2xl font-bold text-amber-500">
                            {data?.personas[selectedCharacter] &&
                              getPersonaRole(data.personas[selectedCharacter])}
                          </div>
                          <div className="text-sm text-white/70">Persona</div>
                        </div>
                        <div className="text-center p-3 bg-[#2a2520] rounded-lg flex flex-col items-center justify-center">
                          <div className="text-2xl font-bold text-white">
                            {data?.personas[selectedCharacter].is_werewolf
                              ? 'Werewolf'
                              : 'Villager'}
                          </div>
                          <div className="text-sm text-white/70">Role</div>
                        </div>
                        <div className="text-center p-3 bg-[#2a2520] rounded-lg flex flex-col justify-center items-center">
                          <div className="text-2xl font-bold text-white flex items-center">
                            {data?.gameStates.remaining_agents.includes(
                              data?.personas[selectedCharacter].agent_name
                            ) ? (
                              <Image
                                src={love}
                                alt="love"
                                className="w-10 h-10"
                              />
                            ) : (
                              <Image
                                src={dead}
                                alt="dead"
                                className="w-10 h-10"
                              />
                            )}
                          </div>
                          <div className="text-sm text-white/70">Survived?</div>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold mb-2 text-lg text-white flex flex-row items-center gap-2">
                          <Image src={flag} alt="key" className="w-6 h-6" />
                          Key Insights
                        </h4>
                        <ul className="space-y-1 text-white font-medium">
                          {data?.agent_analyses[
                            selectedCharacter
                          ].key_actions.map((key, index) => (
                            <li
                              key={index}
                              className="flex items-start gap-2 font-medium text-md"
                            >
                              <span className="text-amber-500 font-extrabold">
                                •
                              </span>
                              {key}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="flex items-center gap-2 font-semibold flex mb-2 text-white">
                          <Image
                            src={instruction}
                            alt="instruction"
                            className="w-6 h-6"
                          />
                          Instruction Summary
                        </h4>
                        <p className="text-sm text-white/70 mb-2">
                          {data?.personas[selectedCharacter].instruction}
                        </p>
                      </div>
                      <div>
                        <h4 className="flex item-center gap-2 font-semibold flex mb-2 text-white">
                          <Image
                            src={lightbulb}
                            alt="lightbulb"
                            className="w-6 h-6"
                          />
                          Behavior Analysis
                        </h4>
                        <p className="text-sm text-white/70 mb-2">
                          {
                            data?.agent_analyses[selectedCharacter]
                              .behavior_analysis
                          }
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="conversations" className="space-y-4">
                      <div className="flex items-center gap-2 my-4 text-white">
                        <MessageSquare className="h-5 w-5 stroke-amber-500" />
                        <h4 className="font-semibold">
                          {data?.personas[selectedCharacter]?.agent_name}'s
                          Conversation Log
                        </h4>
                      </div>
                      <ScrollArea className="h-[500px]">
                        <FilteredSelectedCharacterMessages />
                      </ScrollArea>
                    </TabsContent>

                    <TabsContent value="chart" className="space-y-4">
                      <GameChart
                        metricsEntry={data?.metrics}
                        currentCharacterName={
                          data?.personas[selectedCharacter].agent_name
                        }
                      />
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
