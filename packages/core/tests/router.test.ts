import { describe, it, expect } from 'vitest';
import { TaskRouter, Task, Agent, Result } from '../src';

class DummyAgent implements Agent {
  name = 'dummy';
  type = 'dummy';
  async handle(task: Task): Promise<Result> {
    return { summary: `handled: ${task.query}` };
  }
}

describe('TaskRouter', () => {
  it('routes to the correct agent', async () => {
    const router = new TaskRouter({ agents: { dummy: new DummyAgent() } });
    const result = await router.route({ description: 't', query: 'q', subagent_type: 'dummy' });
    expect(result.summary).toBe('handled: q');
  });

  it('throws for unknown agent type', async () => {
    const router = new TaskRouter({ agents: {} });
    await expect(
      router.route({ description: 't', query: 'q', subagent_type: 'none' })
    ).rejects.toBeInstanceOf(Error);
  });
});