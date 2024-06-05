import express, { Express, Request, Response } from 'express';

import { SimpleBrowser } from './browser';
import config from './config';
import { logger } from './utils';

const session_history: Record<string, SimpleBrowser> = {};

const app: Express = express();

app.use(express.json());

app.post('/', async (req: Request, res: Response) => {
  const {
    session_id,
    action,
  }: {
    session_id: string;
    action: string;
  } = req.body;
  logger.info(`session_id: ${session_id}`);
  logger.info(`action: ${action}`);
  
  if (!session_history[session_id]) {
    session_history[session_id] = new SimpleBrowser();
  }

  const browser = session_history[session_id];

  try {
    res.json(await browser.action(action));
  } catch (err) {
    logger.error(err);
    res.status(400).json(err);
  }
})

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('uncaughtException', e => {
  logger.error(e);
});

const { HOST, PORT } = config;

(async () => {
  app.listen(PORT, HOST, () => {
    logger.info(`⚡️[server]: Server is running at http://${HOST}:${PORT}`);
    try {
      (<any>process).send('ready');
    } catch (err) {}
  });
})();
