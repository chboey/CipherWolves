import Image from 'next/image';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import codebook from '@images/globe.svg';
import magic from '@images/magician.png';
import warlock from '@images/warlock.png';
import warrior from '@images/warrior.png';
import archer from '@images/archer.png';

const characters = [
  {
    id: 1,
    name: 'Deceiver',
    description:
      'Master the Mystical Arts. Wield powerful spells to turn the tide of battle.',
    image: magic,
  },
  {
    id: 2,
    name: 'Analyst',
    description:
      'Embrace Raw Strength. Lead the charge and conquer with sheer power.',
    image: warrior,
  },
  {
    id: 3,
    name: 'Opportunist',
    description:
      'Strike from the Shadows. Swift, cunning, and always one step ahead.',
    image: warlock,
  },
  {
    id: 4,
    name: 'Loyalist',
    description:
      'Sharpshooter Extraordinaire. Precision and agility, from a distance.',
    image: archer,
  },
];

export default function PersonaCard() {
  return (
    <div className="container mx-auto p-10">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
        {characters.map((character) => (
          <Card
            key={character.id}
            className="bg-[#15130A]/60 backdrop-blur-sm border-[#262012]/50 overflow-hidden pt-6 pb-0"
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-center text-amber-100 md:text-xl text-lg font-bold">
                {character.name}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex justify-center h-[14rem] md:h-[22rem]">
              <div className="relative w-40 h-40 md:w-80 md:h-80">
                <Image
                  src={character.image || '/placeholder.svg'}
                  alt={character.name}
                  fill
                  className="rounded-[20px] md:w-[20rem] h-[2rem] w-[18rem] h-[18rem]"
                />
              </div>
            </CardContent>
            <CardFooter className="w-full h-[8rem] pb-0 pt-0 bg-[#15130A] backdrop-blur-sm">
              <p className="text-amber-200/80 text-sm text-center leading-relaxed font-medium">
                {character.description}
              </p>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
