type Level = 'debug' | 'info' | 'warn' | 'error';

const levelOrder: Record<Level, number> = { debug: 10, info: 20, warn: 30, error: 40 };

export class Logger {
  constructor(private minLevel: Level = 'info') {}

  private shouldLog(level: Level) {
    return levelOrder[level] >= levelOrder[this.minLevel];
  }

  debug(message: string, meta?: unknown) {
    if (this.shouldLog('debug')) console.debug('[DEBUG]', message, meta ?? '');
  }
  info(message: string, meta?: unknown) {
    if (this.shouldLog('info')) console.info('[INFO]', message, meta ?? '');
  }
  warn(message: string, meta?: unknown) {
    if (this.shouldLog('warn')) console.warn('[WARN]', message, meta ?? '');
  }
  error(message: string, meta?: unknown) {
    if (this.shouldLog('error')) console.error('[ERROR]', message, meta ?? '');
  }
}