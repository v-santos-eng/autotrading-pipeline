import fs from 'node:fs/promises';
import path from 'node:path';
import { Agent } from '../core/Agent';
import { Task, Result } from '../core/Task';

async function walk(dir: string, files: string[] = []): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await walk(full, files);
    } else {
      files.push(full);
    }
  }
  return files;
}

export class CodeSurveyAgent implements Agent {
  name = 'code-survey';
  description = 'Summarize repository by file types and key files';
  capabilities = ['survey'];

  async run(task: Task): Promise<Result> {
    const base = process.cwd();
    const files = await walk(base);
    const byExt = new Map<string, number>();
    for (const f of files) {
      const ext = path.extname(f) || 'none';
      byExt.set(ext, (byExt.get(ext) ?? 0) + 1);
    }
    const summaryParts = Array.from(byExt.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([ext, count]) => `${ext}:${count}`);
    return { ok: true, summary: summaryParts.join(', '), data: { byExt, files } };
  }
}
