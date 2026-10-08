import { test, expect } from 'vitest';
import { Registry } from '../src/core/Registry';
import { PlanAgent } from '../src/agents/PlanAgent';

test('registers and lists agents', () => {
  const registry = new Registry();
  registry.register(new PlanAgent());
  const list = registry.list();
  expect(list.length).toBe(1);
  expect(list[0].name).toBe('plan');
});