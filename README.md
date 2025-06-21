![CipherWolves Screenshot](https://github.com/user-attachments/assets/ddf94d06-2502-48f4-953e-e219e079b31e)
### Challenge: Automation of Complex Processes
# 🔍 Introduction & Problem Statement

In today's organizations, automating multi-step workflows—like software delivery, business approvals, or incident response—remains a massive challenge.

These processes aren't linear. They require human judgment, real-time debate, shifting priorities, and context that can't be captured by rigid pipelines.

So we asked: How can we automate complex, multi-phase workflows using intelligent agents—while preserving reliability, transparency, and adaptability?

# 💡 Solution: 🐺CipherWolves
![CipherWolves-GameScene](https://github.com/user-attachments/assets/81d5fe87-ef84-4c93-ba6a-c25ce5784201)
Enter CipherWolves—an AI-native platform that simulates human-style decision-making through a team of autonomous, reasoning agents.

We showcase this through a simulation of the social deduction game Werewolf—not for entertainment, but because it captures the ambiguity, conflict, and coordination challenges found in real-world group workflows.

In both settings, agents must operate under uncertainty, adapt to shifting dynamics, and align on high-stakes decisions—even when trust is fractured and goals diverge.

CipherWolves doesn’t simplify this complexity. It embraces it. Each AI agent embodies a distinct reasoning or communication style via personalized personas—not as characters, but as functional archetypes of group behavior.

To drive automation across complex scenarios, agents collaborate through three structured phases:

- Communication – Prompt-driven dialogue surfaces competing observations and strategies. Agents leverage Tavily-powered web search to enrich their reasoning, strategically sourcing external context to strengthen arguments or sow misdirection.

- Analysis – Agents independently interpret conversations, weighing motives and behaviors. Some will challenge the user. Others will target fellow agents. All act with evolving memory and bias.

- Voting – Through tension, persuasion, and shifting alliances, agents converge on a collective decision—sometimes rational, sometimes irrational, but always explainable.

This isn’t just multi-agent automation. This is collaborative reasoning under pressure—a living simulation of how teams argue, align, and act.

What sets CipherWolves apart is its social intelligence: each agent maintains dynamic trust and suspicion scores toward others, influencing every move. These internal metrics evolve in real time, driving alliance-building, deception, and betrayal—mirroring the messy, emotional calculus behind real human decisions.

# 🧠 Post-Game Analysis: Autonomous Reflection at Scale
![PGA](https://github.com/user-attachments/assets/4dbbf17a-6f79-40bd-ac8e-5a7db8ddef1e)
CipherWolves doesn’t just simulate decision-making—it learns from it.

After the final vote is cast, the platform enters a **Post-Analysis Phase**, driven by **Google Gemini**, to perform a deep behavioral autopsy on the game’s execution. This phase turns the simulation into an **analytical artifact**—one that captures not just *what agents did*, but *why they did it*, *how they evolved*, and *what it reveals about collective behavior*.

### 🤖 Powered by Gemini:
Gemini synthesizes every trace—prompt history, communication transcripts, keyword usage, voting behavior, evolving trust/suspicion metrics—into structured insights.

This includes:

- **Agent Profiles**: Breakdown of personality-driven strategy and adaptability
- **Narrative Dominance**: Identification of influence leaders and echo agents
- **Trust Dynamics**: Analysis of how internal metrics shaped alliances and outcomes
- **Manipulation Patterns**: Detection of subtle misdirection or vote engineering
- **Human Interaction Audit**: Evaluation of how the user altered agent behavior

This isn’t generic summarization. It’s **forensic group intelligence analysis**, fully explainable and reproducible.

### 📊 Transparent and Interactive:
Users gain full access to the cognitive blueprint of the match:

- ✅ **Game Transcript**: Full message-by-message log with timestamps
- 📈 **Trust & Suspicion Graphs**: Evolution of interpersonal metrics across rounds
- 🧩 **Voting Rationales**: Agent-level breakdowns of how and why each vote was cast
- 📜 **Behavioral Summaries**: Agent-specific breakdowns of decisions and logic flows


# 🌍 Impact
CipherWolves demonstrates how AI agents can automate and enhance complex, collaborative workflows that traditionally depend on human judgment and negotiation. By simulating nuanced group dynamics, CipherWolves provides a blueprint for:
- Automating decision-making in high-stakes environments (e.g., incident response, business approvals)
- Improving transparency and auditability of group decisions
- Stress-testing organizational processes in a safe, simulated environment
- Accelerating the development of robust, adaptable AI-driven workflows

CipherWolves is not just a game simulation—it's a testbed for the future of collaborative automation.

# 🏗️ Architecture Overview
![Architecture Overview](https://github.com/user-attachments/assets/26b46237-ff18-4ecd-bc53-a18a05fad214)
Under the hood, CipherWolves is engineered for scale, speed, and smart decision-making:

- Uvicorn running FastAPI powers real-time orchestration with Server-Sent Events (SSE), ensuring seamless agent interaction and fast state updates during gameplay.
- Google Gemini drives deep behavioral forensics for our post-analysis system. It analyzes multi-agent conversations, strategic maneuvers, voting patterns, and evolving trust signals—extracting layered insights that mirror real-world decision dynamics.
- Google AI Developer Kit (ADK) enables rapid iteration and low-latency model integration—letting us deploy agent personalities and reasoning styles on the fly.
- Tavily supercharges context awareness by fetching live, real-world data—allowing agents to reference timely, relevant knowledge and adapt strategies accordingly.

Every agent is modular, every phase is documented, and every interaction is trackable.

# 🧠 How We Used the Agent Development Kit
[![ADK Utilisation](https://github.com/user-attachments/assets/66844331-3a6a-4c6c-8419-6ecdfe2b86ff)](https://github.com/user-attachments/assets/66844331-3a6a-4c6c-8419-6ecdfe2b86ff)

CipherWolves runs on the Agent Development Kit (ADK), a modular framework purpose-built for creating socially intelligent, memory-enabled agents. It lets us move beyond stateless LLM calls into a persistent multi-agent environment.

Here's how we engineered it:

- Parent-Child Agent System
    
    We deploy multi-parent, multi-sub-agent clusters mirroring real team dynamics—each with scoped memory, persona, and strategy. Parents delegate cross-agent analysis to sub-agents, who track trust and suspicion in real time.
    
- Function Tool for Web-Augmented Reasoning
    
    Parent agents can use Tavily search through the Function Tool interface—allowing them to discover references, or generate signals during the game. This simulates real-world decision contexts where external data influences discussion flow.
    
- SessionService for Conversation State Management
    
    We use SessionService to retain agent-specific dialogue context across phases—ensuring that when agents shift from discussion to voting or reflection, they preserve alignment and memory continuity.
    
- MemoryService for Long-Term Behavioral Modeling
    
    Trust and suspicion metrics are stored in MemoryService. This long-term store allows agents to develop historical context over the course of the workflow—mirroring how human memory informs judgment over time.

# 🛠️ Technology Stack

### 🧱 Core Frameworks
- **Google ADK (Agent Development Kit)** — Primary agent orchestration framework  
- **FastAPI** — High-performance web API framework  
- **Google Gemini 2.0 Flash** — Post-game analysis & narrative synthesis  
- **Pydantic** — Data validation and serialization  
- **Next.js** — Frontend UI for simulation and post-game analysis

### 📦 Key Dependencies
- `google-adk` — Agent development and orchestration  
- `google-genai` — Google's Generative AI SDK  
- `fastapi` — Web API framework  
- `uvicorn` — ASGI server for FastAPI  
- `python-dotenv` — Environment variable management  
- `faker` — Keyword and agent data generation  


# 🕹️ Short Guidance on Usage

### How to Play:
1. You can experience CipherWolves live at **[cipherwolves.vercel.app](https://cipherwolves.vercel.app)**.
2. **Press “Start Game”** to launch a new simulation.
3. **Observe** as each AI agent takes turns to speak, using limited keywords to debate, bluff, or build alliances.
4. Once **each agent has spoken twice**, it's **your turn** to join the game—decide whether to **deceive** or **help** the agents.
5. The **voting round** follows. If a **majority vote** is achieved, one player (villager or werewolf) is eliminated. Otherwise, a new round begins.
6. The game ends when:
   - The **werewolf is eliminated**, or
   - There are **fewer than 2 villagers left**

### Post-Game Analysis:
After the game, you'll unlock:
- A complete **game log**
- **Dynamic trust and suspicion metrics** between agents
- A **behavioral analysis summary** of each agent’s role and strategy

---

**CipherWolves** - Where AI agents learn to deceive, detect, and dominate in the ultimate social deduction game. 🐺✨ 