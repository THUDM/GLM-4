import winston from 'winston';

import config from './config';

export class TimeoutError extends Error {}

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

export const parseHrtimeToMillisecond = (hrtime: [number, number]): number => {
    return (hrtime[0] + hrtime[1] / 1e9) * 1000;
  };

export const promiseWithTime = <T>(
    promise: Promise<T>
  ): Promise<{
    value: T;
    time: number;
  }> => {
    return new Promise((resolve, reject) => {
      const startTime = process.hrtime();
      promise
        .then(value => {
          resolve({
            value: value,
            time: parseHrtimeToMillisecond(process.hrtime(startTime))
          });
        })
        .catch(err => reject(err));
    });
  };

export const withTimeout = <T>(
    millis: number,
    promise: Promise<T>
  ): Promise<{
    value: T;
    time: number;
  }> => {
    const timeout = new Promise<{ value: T; time: number }>((_, reject) =>
      setTimeout(() => reject(new TimeoutError()), millis)
    );
    return Promise.race([promiseWithTime(promise), timeout]);
  };