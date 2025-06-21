'use client';

import React, { useState } from 'react';
import Image from 'next/image';
import Sword from '@images/sword.svg';
import Torch from '@images/torch.png';
import Key from '@images/medieval-key.svg';
import PersonaCard from '@/components/personasCard';
import cipher from '@images/cipher-logo2.svg';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { createGame } from '@/actions/create-game';

export default function Home() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const handle = async () => {
    setIsLoading(true);

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
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#15130A] overflow-hidden">
      <section className="relative w-screen h-screen overflow-hidden">
        <div className="bg-[url('/bg-game.png')] absolute inset-0 bg-cover bg-center bg-no-repeat shadow-xl mask-b-from-70% mask-t-from-100% mask-b-to-transparent blur-[2px] brightness-[0.5] transform-[1.02]" />

        <div className="relative z-10 h-full flex flex-col items-center justify-center gap-4">
          <div className="text-center space-y-8 relative z-10">
            <h1 className="text-3xl md:text-4xl lg:text-6xl font-bold text-white line-clamp-2 md:w-[55rem] h-[10rem]">
              Decode Alliances, Outsmart Deception, Emerge Victorious
            </h1>
            <h2 className="text-xl md:text-2xl lg:text-4xl font-bold text-white flex items-center justify-center">
              CipherWolves
            </h2>
          </div>

          <Button
            className="bg-[#ffb300] hover:bg-[#FFC400FF] text-gray-900 hover:text-black font-semibold text-lg px-8 py-4 rounded-lg transition-all duration-300 transform hover:scale-108 shadow-md shadow-amber-500/50 flex items-center gap-3 mt-6"
            onClick={handle}
          >
            {isLoading ? 'Starting...' : 'Start Game'}
            <Image src={Sword} alt="sword" priority className="w-6 h-6" />
          </Button>
        </div>
      </section>
      <section className="relative flex flex-row lg:justify-between justify-center w-screen h-[35rem] px-15 my-15">
        <div className="flex flex-col items-top justify-center w-min-fit h-full space-y-6">
          <h1 className="text-2xl lg:text-3xl font-bold text-white line-clamp-2 lg:w-[32rem] lg:text-left text-center">
            Play the Game of Hidden Agendas and Dynamic Alliances!
          </h1>
          <h3 className="text-base lg:text-lg text-white font-light lg:w-[32rem] lg:text-left text-center">
            Dive into CipherWolves, where AI agents navigate coded
            communication, deception, and ever-shifting loyalties. Form
            alliances, decode secrets, and expose imposters in a high-stakes
            battle of strategy and adaptation. In this game, nothing is as it
            seems, and every move tests your wit and cunning.
          </h3>
          <ol className="list-none space-y-3 pl-0 text-white text-base md:text-lg flex flex-col lg:items-start items-center">
            <li className="flex items-start gap-3">
              <Image
                src={Key}
                alt="Check"
                className="w-6 h-6 md:w-8 md:h-8 mt-0.5 flex-shrink-0"
              />
              <span>Dynamic Coded Communication</span>
            </li>
            <li className="flex items-start gap-3">
              <Image
                src={Key}
                alt="Star"
                className="w-6 h-6 md:w-8 md:h-8 mt-0.5 flex-shrink-0"
              />
              <span>Deception and Alliance Strategy</span>
            </li>
            <li className="flex items-start gap-3">
              <Image
                src={Key}
                alt="Arrow"
                className="w-6 h-6 md:w-8 md:h-8 mt-0.5 flex-shrink-0"
              />
              <span>AI-Driven Emotional Intelligence</span>
            </li>
            <li className="flex items-start gap-3">
              <Image
                src={Key}
                alt="Arrow"
                className="w-6 h-6 md:w-8 md:h-8 mt-0.5 flex-shrink-0"
              />
              <span>Interactive Spectator Role</span>
            </li>
          </ol>
          <div className="w-full flex justify-center lg:justify-start">
            <Button
              className="bg-[#ffb300] hover:bg-[#FFC400FF]  text-gray-900 hover:text-black font-semibold text-lg rounded-lg transition-all duration-300 transform hover:scale-108 shadow-[0_2.8px_2.2px_rgba(0,_0,_0,_0.034),_0_6.7px_5.3px_rgba(0,_0,_0,_0.048),_0_12.5px_10px_rgba(0,_0,_0,_0.06),_0_22.3px_17.9px_rgba(0,_0,_0,_0.072),_0_41.8px_33.4px_rgba(0,_0,_0,_0.086),_0_100px_80px_rgba(0,_0,_0,_0.12)] px-10 shadow-amber-500/30"
              onClick={handle}
            >
              Try Demo Now!
              <Image src={Torch} alt="torch" priority className="w-10 h-10" />
            </Button>
          </div>
        </div>
        <div className="grid-cols-2 w-[50rem] h-full justify-center gap-6 text-white text-xl lg:grid hidden">
          <span className="col-span-1 bg-[url('/codebook.png')] lg:bg-cover bg-center w-full h-full rounded-[12px] opacity-85 drop-shadow-orange-400/80 drop-shadow-md" />
          <span className="col-span-1 bg-[url('/deception.png')] lg:bg-cover bg-auto bg-center w-full h-full rounded-[12px] opacity-85 drop-shadow-orange-400/80 drop-shadow-md" />
          <span className="col-span-1 bg-[url('/emotional.png')] lg:bg-cover bg-auto bg-center w-full h-full rounded-[12px] opacity-85 drop-shadow-orange-400/80 drop-shadow-md" />
          <span className="col-span-1 bg-[url('/interactive.png')] lg:bg-cover bg-auto bg-center w-full h-full rounded-[12px] opacity-85 drop-shadow-orange-400/80 drop-shadow-md" />
        </div>
      </section>
      <section className="flex flex-row w-screen h-[5rem]">
        <div className="flex flex-row items-top justify-center w-full h-full my-20">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-white text-center">
            {'Unique Personas and Traits'}
          </h1>
        </div>
      </section>
      <section className="relative w-screen min-h-screen overflow-hidden flex items-center justify-center">
        <div className="bg-[url('/bg2.jpg')] absolute inset-0 bg-cover bg-center bg-no-repeat shadow-xl mask-t-from-70% mask-b-from-100% mask-b-to-transparent blur-[1.5px] brightness-[0.7]" />
        <div className="flex items-center justify-center gap-6 text-white lg:mt-10 mt-20 ">
          <PersonaCard />
        </div>
      </section>
      <section className="relative w-screen h-[7rem] bg-[#262012] brightness-[0.8] overflow-hidden flex items-center justify-center">
        <span className="text-white font-bold text-3xl flex items-center gap-2">
          CipherWolves <Image src={cipher} alt="logo" className="w-12 h-12" />
        </span>
      </section>
    </main>
  );
}
