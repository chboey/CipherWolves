'use server';

export const getConversationHistory = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(
    `${apiDomain}/games/${gameID}/conversation-history`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data;
};

export const getAgentPersonas = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(
    `${apiDomain}/games/${gameID}/get-agent-personas`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data.agent_personas;
};

export const getKeywords = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/games/${gameID}/get-keywords`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data.keywords;
};

export const analyzeConversation = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(
    `${apiDomain}/games/${gameID}/analyze-conversation`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  return data.agent_analyses;
};

export const getDataMetrics = async (gameID: string) => {
  const apiDomain =
    process.env.NEXT_PUBLIC_API_DOMAIN || 'http://localhost:3000';

  const response = await fetch(`${apiDomain}/games/${gameID}/metrics-matrix`, {
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
