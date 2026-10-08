import { Agent } from './Agent';

export class Registry {
  private agents: Map<string, Agent> = new Map();

  register(agent: Agent) {
    this.agents.set(agent.name, agent);
  }

  get(name: string): Agent | undefined {
    return this.agents.get(name);
  }

  list(): Agent[] {
    return Array.from(this.agents.values());
  }
}
