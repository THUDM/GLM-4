import winston from 'winston';

import config from './config';

const logLevel = config.LOG_LEVEL;

export const logger = winston.createLogger({
  level: logLevel,
  format: winston.format.combine(
    winston.format.colorize(),
    winston.format.printf(info => {
      return `${info.level}: ${info.message}`;
    }),
  ),
  transports: [new winston.transports.Console()],
});

console.log('LOG_LEVEL', logLevel);
