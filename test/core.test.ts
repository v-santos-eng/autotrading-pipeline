import { describe, it, expect } from 'vitest'
import { Registry } from '../src/core/Registry'
import { Runner } from '../src/core/Runner'
import { Agent } from '../src/core/Agent'

class EchoAgent implements Agent {
  get name() { return 'echo' }
  get description() { return 'Echoes input' }
  capabilities = ['echo']
  async run(task: { text: string }) { return { ok: true, summary: 'echo', data: task.text } }
}

describe('Registry and Runner', () => {
  it('registers and lists agents', () => {
    const r = new Registry()
    r.register(new EchoAgent())
    expect(r.list().map(a => a.name)).toContain('echo')
  })

  it('runs agent', async () => {
    const r = new Registry()
    r.register(new EchoAgent())
    const runner = new Runner(r)
    const res = await runner.run('echo', { text: 'hello' })
    expect(res.ok).toBe(true)
    expect(res.data).toBe('hello')
  })

  it('fails when agent missing', async () => {
    const r = new Registry()
    const runner = new Runner(r)
    const res = await runner.run('none', { text: 'x' })
    expect(res.ok).toBe(false)
  })
})