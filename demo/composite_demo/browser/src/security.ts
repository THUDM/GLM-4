import { lookup as dnsLookup, promises as dns } from 'node:dns';
import { IncomingHttpHeaders, request as httpRequest } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { isIP } from 'node:net';

import config from './config';

const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

const ipv4ToNumber = (address: string): number => {
  return address.split('.').reduce((value, octet) => (value << 8) + Number(octet), 0) >>> 0;
};

const inIpv4Range = (address: number, base: string, prefix: number): boolean => {
  const mask = prefix === 0 ? 0 : (0xffffffff << (32 - prefix)) >>> 0;
  return (address & mask) === (ipv4ToNumber(base) & mask);
};

const isPublicIpv4 = (address: string): boolean => {
  const value = ipv4ToNumber(address);
  const blocked: Array<[string, number]> = [
    ['0.0.0.0', 8],
    ['10.0.0.0', 8],
    ['100.64.0.0', 10],
    ['127.0.0.0', 8],
    ['169.254.0.0', 16],
    ['172.16.0.0', 12],
    ['192.0.0.0', 24],
    ['192.0.2.0', 24],
    ['192.168.0.0', 16],
    ['198.18.0.0', 15],
    ['198.51.100.0', 24],
    ['203.0.113.0', 24],
    ['224.0.0.0', 4],
    ['240.0.0.0', 4],
  ];
  return !blocked.some(([base, prefix]) => inIpv4Range(value, base, prefix));
};

const isPublicIpv6 = (address: string): boolean => {
  const normalized = address.toLowerCase().split('%', 1)[0];
  if (normalized.startsWith('::ffff:')) {
    const mapped = normalized.slice('::ffff:'.length);
    return isIP(mapped) === 4 && isPublicIpv4(mapped);
  }
  return !(
    normalized === '::' ||
    normalized === '::1' ||
    normalized.startsWith('fc') ||
    normalized.startsWith('fd') ||
    /^fe[89ab]/.test(normalized) ||
    normalized.startsWith('ff') ||
    normalized === '2001:db8::' ||
    normalized.startsWith('2001:db8:')
  );
};

const isPublicAddress = (address: string): boolean => {
  const family = isIP(address);
  if (family === 4) return isPublicIpv4(address);
  if (family === 6) return isPublicIpv6(address);
  return false;
};

export const validatePublicUrl = async (rawUrl: string): Promise<URL> => {
  if (rawUrl.length > config.MAX_URL_LENGTH) throw new Error('URL is too long');

  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error('Invalid URL');
  }
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Only HTTP(S) URLs are allowed');
  if (parsed.username || parsed.password) throw new Error('Credentials in URLs are not allowed');

  const hostname = parsed.hostname.replace(/^\[|\]$/g, '').replace(/\.$/, '').toLowerCase();
  if (!hostname || hostname === 'localhost' || hostname.endsWith('.localhost') || hostname.endsWith('.local')) {
    throw new Error('Local hostnames are not allowed');
  }
  if (config.ALLOW_PRIVATE_NETWORKS) return parsed;

  const literalFamily = isIP(hostname);
  if (literalFamily && !isPublicAddress(hostname)) throw new Error('Private and special-use addresses are not allowed');
  if (!literalFamily) {
    let addresses: Array<{ address: string }>;
    try {
      addresses = await dns.lookup(hostname, { all: true, verbatim: true });
    } catch {
      throw new Error('Unable to resolve URL hostname');
    }
    if (addresses.length === 0 || addresses.some(({ address }) => !isPublicAddress(address))) {
      throw new Error('Hostname resolves to a private or special-use address');
    }
  }
  return parsed;
};

export interface FetchTextResult {
  text: string;
  time: number;
  finalUrl: string;
}

interface SafeRequestInit {
  headers?: Record<string, string>;
}

interface RawResponse {
  status: number;
  headers: IncomingHttpHeaders;
  text: string;
}

const safeLookup = (hostname: string, _options: unknown, callback: (...args: any[]) => void): void => {
  dnsLookup(hostname, { all: true, verbatim: true }, (error, addresses) => {
    if (error) {
      callback(error);
      return;
    }
    if (addresses.length === 0 || addresses.some(({ address }) => !isPublicAddress(address))) {
      callback(new Error('Hostname resolved to a private or special-use address'));
      return;
    }
    const selected = addresses[0];
    callback(null, selected.address, selected.family);
  });
};

const requestText = (url: URL, init: SafeRequestInit, signal: AbortSignal): Promise<RawResponse> => {
  return new Promise((resolve, reject) => {
    const request = (url.protocol === 'https:' ? httpsRequest : httpRequest)(
      url,
      {
        method: 'GET',
        headers: {
          Accept: 'text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.1',
          'User-Agent': 'GLM-4-browser/1.0',
          ...init.headers,
        },
        lookup: config.ALLOW_PRIVATE_NETWORKS ? undefined : safeLookup,
        signal,
      },
      response => {
        const status = response.statusCode ?? 0;
        if (REDIRECT_STATUSES.has(status)) {
          response.resume();
          resolve({ status, headers: response.headers, text: '' });
          return;
        }
        if (status < 200 || status >= 300) {
          response.resume();
          reject(new Error(`Upstream returned HTTP ${status}`));
          return;
        }

        const contentLength = Number.parseInt(response.headers['content-length'] ?? '', 10);
        if (Number.isFinite(contentLength) && contentLength > config.MAX_RESPONSE_BYTES) {
          response.destroy();
          reject(new Error('Upstream response is too large'));
          return;
        }

        const chunks: Buffer[] = [];
        let total = 0;
        response.on('data', (chunk: Buffer) => {
          total += chunk.length;
          if (total > config.MAX_RESPONSE_BYTES) {
            response.destroy(new Error('Upstream response is too large'));
            return;
          }
          chunks.push(chunk);
        });
        response.on('end', () => {
          resolve({ status, headers: response.headers, text: Buffer.concat(chunks).toString('utf8') });
        });
        response.on('error', reject);
      },
    );
    request.on('error', reject);
    request.end();
  });
};

export const fetchTextWithLimits = async (
  rawUrl: string,
  init: SafeRequestInit = {},
): Promise<FetchTextResult> => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.BROWSER_TIMEOUT);
  const started = process.hrtime.bigint();
  let currentUrl = rawUrl;

  try {
    for (let redirectCount = 0; redirectCount <= config.MAX_REDIRECTS; redirectCount++) {
      const parsed = await validatePublicUrl(currentUrl);
      const response = await requestText(parsed, init, controller.signal);

      if (REDIRECT_STATUSES.has(response.status)) {
        if (redirectCount >= config.MAX_REDIRECTS) throw new Error('Too many redirects');
        const location = response.headers.location;
        if (!location) throw new Error('Redirect response has no Location header');
        currentUrl = new URL(location, currentUrl).href;
        continue;
      }
      const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
      return { text: response.text, time: elapsed, finalUrl: currentUrl };
    }
  } catch (error) {
    if (error instanceof Error && (error.name === 'AbortError' || error.name === 'TimeoutError')) {
      throw new Error('Upstream request timed out');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
  throw new Error('Unable to fetch URL');
};
