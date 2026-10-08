import { Agent, Result, Task } from '@agents/core';

export class SearchAgent implements Agent {
  name = 'search';
  type = 'search';

  async handle(task: Task): Promise<Result> {
    const summary = `Search executed for query: ${task.query}`;
    const details = {
      description: task.description,
      response_language: task.response_language ?? 'en'
    };
    return { summary, details };
  }
}