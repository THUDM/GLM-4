import assert from 'node:assert/strict';
import http from 'node:http';
import { AddressInfo } from 'node:net';
import test from 'node:test';

import config from './config';
import { fetchTextWithLimits, validatePublicUrl } from './security';

test('URL and fetch security boundaries', async () => {
  for (const url of [
    'http://127.0.0.1/',
    'http://[::1]/',
    'http://169.254.169.254/latest/meta-data/',
    'file:///etc/passwd',
    'http://user:pass@example.com/',
  ]) {
    await assert.rejects(validatePublicUrl(url));
  }

  const previous = {
    allowPrivateNetworks: config.ALLOW_PRIVATE_NETWORKS,
    maxResponseBytes: config.MAX_RESPONSE_BYTES,
    maxRedirects: config.MAX_REDIRECTS,
    timeout: config.BROWSER_TIMEOUT,
  };
  config.ALLOW_PRIVATE_NETWORKS = true;
  config.MAX_RESPONSE_BYTES = 32;
  config.MAX_REDIRECTS = 1;
  config.BROWSER_TIMEOUT = 200;

  const server = http.createServer((request, response) => {
    if (request.url === '/ok') return response.end('ok');
    if (request.url === '/redirect') {
      response.writeHead(302, { Location: '/ok' });
      return response.end();
    }
    if (request.url === '/loop') {
      response.writeHead(302, { Location: '/loop' });
      return response.end();
    }
    if (request.url === '/declared') {
      response.writeHead(200, { 'Content-Length': '64' });
      return response.end('x'.repeat(64));
    }
    if (request.url === '/chunked') {
      response.write('x'.repeat(24));
      return response.end('x'.repeat(24));
    }
    if (request.url === '/slow') return;
    response.writeHead(404);
    return response.end();
  });

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    assert.equal((await fetchTextWithLimits(`${baseUrl}/redirect`)).text, 'ok');
    for (const path of ['/loop', '/declared', '/chunked', '/slow']) {
      await assert.rejects(fetchTextWithLimits(baseUrl + path));
    }
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    );
    config.ALLOW_PRIVATE_NETWORKS = previous.allowPrivateNetworks;
    config.MAX_RESPONSE_BYTES = previous.maxResponseBytes;
    config.MAX_REDIRECTS = previous.maxRedirects;
    config.BROWSER_TIMEOUT = previous.timeout;
  }
});
