export interface Task {
  description: string;
  query: string;
  subagent_type: string;
  response_language?: string;
}

export interface Result {
  summary: string;
  details?: unknown;
}

export interface Agent {
  name: string;
  type: string;
  handle(task: Task): Promise<Result>;
}

export interface RouterOptions {
  agents: Record<string, Agent>;
}