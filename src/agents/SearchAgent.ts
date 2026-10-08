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

export class SearchAgent implements Agent {
  name = 'search';
  description = 'Search files by name and content in the current workspace';
  capabilities = ['search'];

  async run(task: Task): Promise<Result> {
    const query = String(task.params?.query ?? '').toLowerCase();
    const base = process.cwd();
    const allFiles = await walk(base);
    const matched: string[] = [];
    for (const f of allFiles) {
      const name = path.basename(f).toLowerCase();
      if (query && name.includes(query)) {
        matched.push(f);
        continue;
      }
      if (query) {
        try {
          const content = await fs.readFile(f, 'utf8');
          if (content.toLowerCase().includes(query)) matched.push(f);
        } catch {}
      }
    }
    return { ok: true, summary: `Found ${matched.length} files`, data: matched };
  }
}
