import { Agent } from '../core/Agent';
import { Task, Result } from '../core/Task';

export class PlanAgent implements Agent {
  name = 'plan';
  description = 'Generate a simple step-by-step plan for a task';
  capabilities = ['plan'];

  async run(task: Task): Promise<Result> {
    const goal = task.description;
    const steps = [
      'Understand requirements',
      'Determine stack and constraints',
      'Design API and modules',
      'Implement features iteratively',
      'Test and validate outputs'
    ];
    return { ok: true, summary: goal, data: steps };
  }
}
