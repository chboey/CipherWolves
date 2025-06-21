'use server';

export const createGame = async () => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/create_game`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      num_agents: 5,
      model: 'gemini-2.0-flash-001',
      num_keywords: 300,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data;
};
