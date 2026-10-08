import { Command } from 'commander';
import { Registry } from '../core/Registry.js';
import { Runner } from '../core/Runner.js';
import { Task } from '../core/Task.js';
import { SearchAgent } from '../agents/SearchAgent.js';
import { CodeSurveyAgent } from '../agents/CodeSurveyAgent.js';
import { PlanAgent } from '../agents/PlanAgent.js';
import { loadConfig } from '../config/index.js';

async function makeRegistry() {
  const registry = new Registry();
  registry.register(new SearchAgent());
  registry.register(new CodeSurveyAgent());
  registry.register(new PlanAgent());
  return registry;
}

async function main() {
  const program = new Command();
  program.name('agents').description('Run and manage local agents').version('0.1.0');

  program
    .command('list')
    .description('List available agents')
    .action(async () => {
      const registry = await makeRegistry();
      const agents = registry.list();
      for (const a of agents) {
        console.log(`${a.name} - ${a.description}`);
      }
    });

  program
    .command('run')
    .description('Run an agent with a task')
    .requiredOption('-a, --agent <name>', 'Agent name')
    .requiredOption('-t, --task <description>', 'Task description')
    .option('-p, --params <json>', 'Task params as JSON')
    .option('-c, --config <path>', 'Config file path')
    .action(async (opts) => {
      const registry = await makeRegistry();
      const runner = new Runner(registry);
      const config = await loadConfig(opts.config);
      const params = opts.params ? JSON.parse(opts.params) : {};
      const now = new Date();
      const task: Task = {
        id: `${now.getTime()}`,
        description: opts.task,
        params: { ...params, text: opts.task },
        createdAt: now
      };
      const result = await runner.run(opts.agent, task);
      if (!result.ok) {
        console.error('Agent failed');
      }
      console.log(JSON.stringify({ result, config }, null, 2));
      process.exit(result.ok ? 0 : 1);
    });

  await program.parseAsync(process.argv);
}

main();