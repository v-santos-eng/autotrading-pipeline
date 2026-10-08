import { describe, it, expect } from 'vitest';
import { SearchAgent } from '../src';

describe('SearchAgent', () => {
  it('returns a summary including the query', async () => {
    const agent = new SearchAgent();
    const result = await agent.handle({
      description: 'd',
      query: 'find-this',
      subagent_type: 'search',
      response_language: 'en'
    });
    expect(result.summary).toContain('find-this');
  });
});