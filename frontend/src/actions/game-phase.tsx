'use server';

export const StartGameRound = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/games/${gameID}/round`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
};

export const startVoting = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/games/${gameID}/vote`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
};

export const startAnalysis = async (gameID: string, input: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/games/${gameID}/analysis`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: input,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
};

export const checkGameState = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/games/${gameID}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data;
};
