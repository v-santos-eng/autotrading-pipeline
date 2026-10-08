import { Task, Result } from './Task';

export interface Agent {
  name: string;
  description: string;
  capabilities: string[];
  run(task: Task): Promise<Result>;
}
