'use client';

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Eye } from 'lucide-react';
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
} from '@/components/ui/chart';
import { PieChart, Pie, Cell } from 'recharts';

interface TrustSuspicion {
  [agent: string]: {
    trust_level: number | null;
    suspicion_level: number | null;
  };
}

// Type definitions
export interface AgentMetrics {
  round: number;
  phase: string;
  analyzer: string;
  trust_suspicion: TrustSuspicion;
}

interface ProcessedMetrics {
  suspicion_from_others: Record<string, number>;
  trust_from_others: Record<string, number>;
}

const chartConfig = {
  suspicion: {
    label: 'Suspicion Level',
    color: 'hsl(var(--chart-1))',
  },
  trust: {
    label: 'Trust Level',
    color: 'hsl(var(--chart-2))',
  },
} satisfies ChartConfig;

const characterColors = {
  agent_Alice: '#F54DA8',
  agent_Bob: '#832121',
  agent_Charlie: '#E21414',
  agent_Dom: '#FFE050',
  agent_Elise: '#8138FEFF',
};

interface GameChartProps {
  metricsEntry: AgentMetrics[] | undefined;
  currentCharacterName: string | undefined;
}

export default function GameChart({
  metricsEntry,
  currentCharacterName,
}: GameChartProps) {
  const [selectedRound, setSelectedRound] = useState(1);
  const [selectedPhase, setSelectedPhase] = useState('communication');
  const [isAnimating, setIsAnimating] = useState(false);

  if (!currentCharacterName || !metricsEntry) {
    return;
  }

  const processMetrics = (
    targetAgent: string,
    round: number,
    phase: string
  ): ProcessedMetrics => {
    const relevantMetrics = metricsEntry?.filter((entry) => {
      const analyzerAgent = entry.analyzer.split('_').slice(0, 2).join('_');
      return (
        entry.round === round &&
        entry.phase === phase &&
        analyzerAgent !== targetAgent
      );
    });

    const suspicion_from_others: Record<string, number> = {};
    const trust_from_others: Record<string, number> = {};

    relevantMetrics?.forEach((entry) => {
      const analyzerAgent = entry.analyzer.split('_').slice(0, 2).join('_');

      const trustSuspicionData = entry.trust_suspicion[targetAgent];

      if (trustSuspicionData) {
        if (trustSuspicionData.suspicion_level !== null) {
          suspicion_from_others[analyzerAgent] =
            trustSuspicionData.suspicion_level;
        }
        if (trustSuspicionData.trust_level !== null) {
          trust_from_others[analyzerAgent] = trustSuspicionData.trust_level;
        }
      }
    });

    return { suspicion_from_others, trust_from_others };
  };

  const availableRounds = [
    ...new Set(metricsEntry?.map((entry) => entry.round)),
  ].sort();
  const availablePhases = [
    ...new Set(metricsEntry?.map((entry) => entry.phase)),
  ].filter((phase) => phase !== 'analysis');

  const currentMetrics = processMetrics(
    currentCharacterName,
    selectedRound,
    selectedPhase
  );

  // Prepare suspicion chart data
  const suspicionData = Object.entries(
    currentMetrics.suspicion_from_others
  ).map(([name, value]) => ({
    name: name.split('_')[1] || name, // Use first name only
    fullName: name,
    value: Math.round((value as number) * 100), // Convert 0-1 to percentage
    fill: characterColors[name as keyof typeof characterColors],
  }));

  // Prepare trust chart data
  const trustData = Object.entries(currentMetrics.trust_from_others).map(
    ([name, value]) => ({
      name: name.split('_')[1] || name, // Use first name only
      fullName: name,
      value: Math.round((value as number) * 100), // Convert 0-1 to percentage
      fill: characterColors[name as keyof typeof characterColors],
    })
  );

  const handleRoundChange = (round: number) => {
    if (round !== selectedRound) {
      setIsAnimating(true);
      setTimeout(() => {
        setSelectedRound(round);
        setIsAnimating(false);
      }, 150);
    }
  };

  const handlePhaseChange = (phase: string) => {
    if (phase !== selectedPhase) {
      setIsAnimating(true);
      setTimeout(() => {
        setSelectedPhase(phase);
        setIsAnimating(false);
      }, 150);
    }
  };

  const avgSuspicion =
    suspicionData.length > 0
      ? Math.round(
          suspicionData.reduce((sum, item) => sum + item.value, 0) /
            suspicionData.length
        )
      : 0;

  const avgTrust =
    trustData.length > 0
      ? Math.round(
          trustData.reduce((sum, item) => sum + item.value, 0) /
            trustData.length
        )
      : 0;

  return (
    <div>
      <Card className="border-0 bg-[#1a1810]">
        <CardContent className="px-0">
          {/* Metrics Section */}
          <div className="space-y-6">
            <div className="border-t border-white/10 pt-6">
              <h4 className="font-semibold flex items-center gap-2 mb-4 text-white">
                <Eye className="h-5 w-5 text-amber-500" />
                How Others Perceive {currentCharacterName.split('_')[1]}
              </h4>

              {/* Round Selection */}
              <div className="mb-4">
                <h5 className="text-white font-medium mb-3">Select Round</h5>
                <div className="flex gap-2">
                  {availableRounds.map((round) => (
                    <Button
                      key={round}
                      onClick={() => handleRoundChange(round)}
                      variant={selectedRound === round ? 'default' : 'outline'}
                      className={`${
                        selectedRound === round
                          ? 'bg-amber-500 text-black hover:bg-amber-600'
                          : 'bg-[#2a2520] text-white border-white/20 hover:bg-[#3a3530]'
                      }`}
                    >
                      Round {round}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Phase Selection */}
              <div className="mb-6">
                <h5 className="text-white font-medium mb-3">Select Phase</h5>
                <div className="flex gap-2 flex-wrap">
                  {availablePhases.map((phase) => (
                    <Button
                      key={phase}
                      onClick={() => handlePhaseChange(phase)}
                      variant={selectedPhase === phase ? 'default' : 'outline'}
                      className={`${
                        selectedPhase === phase
                          ? 'bg-amber-500 text-black hover:bg-amber-600'
                          : 'bg-[#2a2520] text-white border-white/20 hover:bg-[#3a3530]'
                      }`}
                    >
                      {phase.charAt(0).toUpperCase() + phase.slice(1)}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Current Selection Display */}
              <div className="mb-6 p-4 bg-[#2a2520] rounded-lg">
                <div className="flex items-center justify-between">
                  <h5 className="text-white font-medium">
                    Round {selectedRound} -{' '}
                    {selectedPhase.charAt(0).toUpperCase() +
                      selectedPhase.slice(1)}{' '}
                    Phase
                  </h5>
                  <div className="flex gap-2">
                    <Badge
                      variant="outline"
                      className="text-red-400 border-red-400/20"
                    >
                      Avg Suspicion: {avgSuspicion}%
                    </Badge>
                    <Badge
                      variant="outline"
                      className="text-green-400 border-green-400/20"
                    >
                      Avg Trust: {avgTrust}%
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Dual Donut Charts */}
              <div
                className={`grid grid-cols-1 lg:grid-cols-2 gap-6 transition-opacity duration-300 ${
                  isAnimating ? 'opacity-50' : 'opacity-100'
                }`}
              >
                {/* Suspicion Chart */}
                <div className="bg-[#2a2520] rounded-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h5 className="text-white font-medium flex items-center gap-2">
                      <TrendingDown className="h-4 w-4 text-red-500" />
                      Suspicion Levels
                    </h5>
                  </div>

                  {suspicionData.length > 0 ? (
                    <>
                      <ChartContainer
                        config={chartConfig}
                        className="mx-auto aspect-square max-h-[300px]"
                      >
                        <PieChart>
                          <ChartTooltip
                            cursor={false}
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const data = payload[0].payload;
                                return (
                                  <div className="bg-[#1a1810] p-3 rounded-lg border border-red-500/20">
                                    <p className="text-white font-medium">
                                      {data.fullName.split('_')[1]}
                                    </p>
                                    <p className="text-red-400">
                                      Suspicion: {data.value}%
                                    </p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Pie
                            data={suspicionData}
                            dataKey="value"
                            nameKey="name"
                            innerRadius={60}
                            outerRadius={110}
                          >
                            {suspicionData.map((entry, index) => (
                              <Cell
                                key={`suspicion-cell-${index}`}
                                fill={entry.fill}
                              />
                            ))}
                          </Pie>
                          <text
                            x="50%"
                            y="45%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="fill-white text-xl font-bold"
                          >
                            {avgSuspicion}%
                          </text>
                          <text
                            x="50%"
                            y="55%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="fill-white/70 text-xs"
                          >
                            Avg Suspicion
                          </text>
                        </PieChart>
                      </ChartContainer>

                      {/* Suspicion Legend */}
                      <div className="grid grid-cols-2 gap-2 mt-4">
                        {suspicionData.map((item) => (
                          <div
                            key={`suspicion-${item.name}`}
                            className="flex items-center gap-2"
                          >
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: item.fill }}
                            />
                            <span className="text-white text-sm">
                              {item.name}: {item.value}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-white/50">
                      No suspicion data available
                    </div>
                  )}
                </div>

                {/* Trust Chart */}
                <div className="bg-[#2a2520] rounded-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h5 className="text-white font-medium flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-green-500" />
                      Trust Levels
                    </h5>
                  </div>

                  {trustData.length > 0 ? (
                    <>
                      <ChartContainer
                        config={chartConfig}
                        className="mx-auto aspect-square max-h-[300px]"
                      >
                        <PieChart>
                          <ChartTooltip
                            cursor={false}
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const data = payload[0].payload;
                                return (
                                  <div className="bg-[#1a1810] p-3 rounded-lg border border-green-500/20">
                                    <p className="text-white font-medium">
                                      {data.fullName.split('_')[1]}
                                    </p>
                                    <p className="text-green-400">
                                      Trust: {data.value}%
                                    </p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Pie
                            data={trustData}
                            dataKey="value"
                            nameKey="name"
                            innerRadius={60}
                            outerRadius={110}
                          >
                            {trustData.map((entry, index) => (
                              <Cell
                                key={`trust-cell-${index}`}
                                fill={entry.fill}
                              />
                            ))}
                          </Pie>
                          <text
                            x="50%"
                            y="45%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="fill-white text-xl font-bold"
                          >
                            {avgTrust}%
                          </text>
                          <text
                            x="50%"
                            y="55%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="fill-white/70 text-xs"
                          >
                            Avg Trust
                          </text>
                        </PieChart>
                      </ChartContainer>

                      {/* Trust Legend */}
                      <div className="grid grid-cols-2 gap-2 mt-4">
                        {trustData.map((item) => (
                          <div
                            key={`trust-${item.name}`}
                            className="flex items-center gap-2"
                          >
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: item.fill }}
                            />
                            <span className="text-white text-sm">
                              {item.name}: {item.value}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-[300px] text-white/50">
                      No trust data available
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
