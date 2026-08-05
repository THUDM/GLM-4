const intFromEnv = (name: string, fallback: number): number => {
    const value = Number.parseInt(process.env[name] ?? '', 10);
    return Number.isFinite(value) && value > 0 ? value : fallback;
};

export default {
    LOG_LEVEL: process.env.BROWSER_LOG_LEVEL ?? 'info',
    BROWSER_TIMEOUT: intFromEnv('BROWSER_TIMEOUT_MS', 10000),
    MAX_RESPONSE_BYTES: intFromEnv('BROWSER_MAX_RESPONSE_BYTES', 512 * 1024),
    MAX_REDIRECTS: intFromEnv('BROWSER_MAX_REDIRECTS', 3),
    MAX_URL_LENGTH: intFromEnv('BROWSER_MAX_URL_LENGTH', 4096),
    MAX_ACTION_BYTES: intFromEnv('BROWSER_MAX_ACTION_BYTES', 16 * 1024),
    MAX_ACTION_LINES: intFromEnv('BROWSER_MAX_ACTION_LINES', 10),
    MAX_MCLICK_IDS: intFromEnv('BROWSER_MAX_MCLICK_IDS', 3),
    MAX_CONCURRENT_REQUESTS: intFromEnv('BROWSER_MAX_CONCURRENT_REQUESTS', 4),
    MAX_REQUESTS_PER_MINUTE: intFromEnv('BROWSER_MAX_REQUESTS_PER_MINUTE', 120),
    MAX_RATE_COUNTERS: intFromEnv('BROWSER_MAX_RATE_COUNTERS', 1024),
    MAX_SESSIONS: intFromEnv('BROWSER_MAX_SESSIONS', 20),
    SESSION_TTL_MS: intFromEnv('BROWSER_SESSION_TTL_MS', 30 * 60 * 1000),
    MAX_PAGE_STACK: intFromEnv('BROWSER_MAX_PAGE_STACK', 5),
    MAX_QUOTES: intFromEnv('BROWSER_MAX_QUOTES', 20),
    ALLOW_PRIVATE_NETWORKS: process.env.BROWSER_ALLOW_PRIVATE_NETWORKS === '1',
    API_KEY: process.env.BROWSER_API_KEY ?? '',
    BING_SEARCH_API_URL: 'https://api.bing.microsoft.com/v7.0/custom/',
    BING_SEARCH_API_KEY: process.env.BING_SEARCH_API_KEY ?? '',
    CUSTOM_CONFIG_ID: process.env.BING_CUSTOM_CONFIG_ID ?? '',
    HOST: process.env.BROWSER_HOST ?? '127.0.0.1',
    PORT: intFromEnv('BROWSER_PORT', 3000),
};
