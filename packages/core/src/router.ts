import { Agent, Result, RouterOptions, Task } from './types';
import { Logger } from './logger';

export class TaskRouter {
  private readonly logger = new Logger('info');
  private readonly agents: Record<string, Agent>;

  constructor(options: RouterOptions) {
    this.agents = options.agents;
  }

  async route(task: Task): Promise<Result> {
    this.logger.info('Routing task', { task });
    const agent = this.agents[task.subagent_type];
    if (!agent) {
      this.logger.error('No agent found for type', { type: task.subagent_type });
      throw new Error(`Unknown agent type: ${task.subagent_type}`);
    }
    try {
      const result = await agent.handle(task);
      this.logger.info('Task handled by agent', { agent: agent.name });
      return result;
    } catch (err) {
      this.logger.error('Agent execution failed', err);
      throw err instanceof Error ? err : new Error('Agent execution failed');
    }
  }
}