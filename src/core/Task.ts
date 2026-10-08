export type TaskParams = Record<string, unknown>;

export interface Task {
  id: string;
  description: string;
  params?: TaskParams;
  createdAt: Date;
}

export interface Result {
  ok: boolean;
  summary: string;
  data?: unknown;
}
