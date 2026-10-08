import { Registry } from './Registry';
import { Task, Result } from './Task';

export class Runner {
  constructor(private registry: Registry) {}

  async run(agentName: string, task: Task): Promise<Result> {
    const agent = this.registry.get(agentName);
    if (!agent) {
      return { ok: false, summary: `Agent not found: ${agentName}` };
    }
    try {
      return await agent.run(task);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      return { ok: false, summary: message };
    }
  }
}
