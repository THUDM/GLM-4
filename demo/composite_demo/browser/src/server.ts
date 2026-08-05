import express, { Express, Request, Response } from 'express';
import { timingSafeEqual } from 'node:crypto';

import { SimpleBrowser } from './browser';
import config from './config';
import { logger } from './utils';

interface SessionEntry {
  browser: SimpleBrowser;
  lastAccess: number;
  busy: boolean;
}

const sessionHistory = new Map<string, SessionEntry>();
const requestCounters = new Map<string, { count: number; windowStarted: number }>();
let activeRequests = 0;

const app: Express = express();

app.disable('x-powered-by');
app.use(express.json({ limit: config.MAX_ACTION_BYTES + 1024 }));

app.use((req: Request, res: Response, next) => {
  if (!config.API_KEY) {
    next();
    return;
  }
  const supplied = Buffer.from(req.header('authorization') ?? '');
  const expected = Buffer.from(`Bearer ${config.API_KEY}`);
  if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }
  next();
});

app.use((req: Request, res: Response, next) => {
  const now = Date.now();
  const key = req.socket.remoteAddress ?? 'unknown';
  if (requestCounters.size >= config.MAX_RATE_COUNTERS) {
    for (const [address, value] of requestCounters.entries()) {
      if (now - value.windowStarted >= 60_000) requestCounters.delete(address);
    }
    while (!requestCounters.has(key) && requestCounters.size >= config.MAX_RATE_COUNTERS) {
      const oldest = [...requestCounters.entries()].sort((a, b) => a[1].windowStarted - b[1].windowStarted)[0];
      if (!oldest) break;
      requestCounters.delete(oldest[0]);
    }
  }
  let counter = requestCounters.get(key);
  if (!counter || now - counter.windowStarted >= 60_000) {
    counter = { count: 0, windowStarted: now };
    requestCounters.set(key, counter);
  }
  counter.count += 1;
  if (counter.count > config.MAX_REQUESTS_PER_MINUTE) {
    res.status(429).json({ error: 'Too many requests' });
    return;
  }
  next();
});

const cleanupSessions = (now: number): void => {
  for (const [id, entry] of sessionHistory.entries()) {
    if (!entry.busy && now - entry.lastAccess > config.SESSION_TTL_MS) sessionHistory.delete(id);
  }
};

const ensureSessionCapacity = (): boolean => {
  while (sessionHistory.size >= config.MAX_SESSIONS) {
    const oldest = [...sessionHistory.entries()]
      .filter(([, entry]) => !entry.busy)
      .sort((a, b) => a[1].lastAccess - b[1].lastAccess)[0];
    if (!oldest) return false;
    sessionHistory.delete(oldest[0]);
  }
  return true;
};

app.post('/', async (req: Request, res: Response) => {
  if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body)) {
    res.status(400).json({ error: 'JSON object body required' });
    return;
  }
  const {
    session_id,
    action,
  }: {
    session_id: string;
    action: string;
  } = req.body;
  if (typeof session_id !== 'string' || !/^[A-Za-z0-9._:-]{1,128}$/.test(session_id)) {
    res.status(400).json({ error: 'Invalid session_id' });
    return;
  }
  if (typeof action !== 'string' || Buffer.byteLength(action, 'utf8') > config.MAX_ACTION_BYTES) {
    res.status(400).json({ error: 'Invalid action' });
    return;
  }
  if (activeRequests >= config.MAX_CONCURRENT_REQUESTS) {
    res.status(503).json({ error: 'Browser service is busy' });
    return;
  }

  logger.info(`session_id: ${session_id}; action_bytes: ${Buffer.byteLength(action, 'utf8')}`);
  const now = Date.now();
  cleanupSessions(now);

  let session = sessionHistory.get(session_id);
  if (!session) {
    if (!ensureSessionCapacity()) {
      res.status(503).json({ error: 'All browser sessions are busy' });
      return;
    }
    session = { browser: new SimpleBrowser(), lastAccess: now, busy: false };
    sessionHistory.set(session_id, session);
  }
  session.lastAccess = now;
  if (session.busy) {
    res.status(409).json({ error: 'Another action is already running for this session' });
    return;
  }

  activeRequests += 1;
  session.busy = true;

  try {
    res.json(await session.browser.action(action));
  } catch (err) {
    logger.error(err);
    res.status(400).json({ error: err instanceof Error ? err.message : 'Browser action failed' });
  } finally {
    activeRequests -= 1;
    session.busy = false;
  }
});

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('uncaughtException', e => {
  logger.error(e);
  process.exit(1);
});

const { HOST, PORT } = config;

if (!['127.0.0.1', '::1', 'localhost'].includes(HOST) && !config.API_KEY) {
  throw new Error('BROWSER_API_KEY is required when binding the browser service to a non-loopback address');
}

(async () => {
  app.listen(PORT, HOST, () => {
    logger.info(`⚡️[server]: Server is running at http://${HOST}:${PORT}`);
    try {
      (<any>process).send('ready');
    } catch (err) {}
  });
})();
