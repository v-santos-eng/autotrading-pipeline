import { TaskRouter, Task } from '@agents/core';
import { SearchAgent } from '@agents/search';

function parseArgs(argv: string[]) {
  const args: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
      args[key] = val;
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const subagent = args.subagent ?? 'search';
  const query = args.query ?? 'hello world';
  const response_language = args.lang ?? 'en';

  const router = new TaskRouter({ agents: { search: new SearchAgent() } });
  const task: Task = {
    description: 'Run task via CLI',
    query,
    subagent_type: subagent,
    response_language
  };

  try {
    const result = await router.route(task);
    console.log('Summary:', result.summary);
    if (result.details) console.log('Details:', JSON.stringify(result.details, null, 2));
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('Error:', msg);
    process.exitCode = 1;
  }
}

main();