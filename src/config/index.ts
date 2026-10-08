import fs from 'node:fs/promises';
import path from 'node:path';


export type AgentsConfig = {
  agents?: string[];
};

const defaultConfig: AgentsConfig = {};

export async function loadConfig(configPath?: string): Promise<AgentsConfig> {
  const resolved = configPath
    ? path.resolve(process.cwd(), configPath)
    : path.resolve(process.cwd(), 'agents.config.json');
  try {
    const content = await fs.readFile(resolved, 'utf8');
    if (resolved.endsWith('.yaml') || resolved.endsWith('.yml')) {
      // @ts-ignore
      const { parse } = await import('yaml');
      return { ...defaultConfig, ...parse(content) };
    }
    return { ...defaultConfig, ...JSON.parse(content) };
  } catch {
    return defaultConfig;
  }
}