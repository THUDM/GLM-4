import { JSDOM } from 'jsdom';
import TurndownService from 'turndown';

import config from './config';
import { Message, ToolObservation } from './types';
import { logger, withTimeout } from './utils';

// represent a quote from a display
interface Quote {
  text: string;
  metadata: Metadata[];
}

interface ActionResult {
  contentType: string;
  metadataList?: TetherQuoteMetadata[];
  metadata?: any;
  roleMetadata: string;
  message: string;
}

// represent a piece of metadata to be marked in the final answer
interface Metadata {
  type: string;
  title: string;
  url: string;
  lines: string[];
}

interface TetherQuoteExtra {
  cited_message_idx: number;
  evidence_text: string;
}

interface TetherQuoteMetadata {
  type: string;
  title: string;
  url: string;
  text: string;
  pub_date?: string;
  extra?: TetherQuoteExtra;
}

interface Citation {
  citation_format_type: string;
  start_ix: number;
  end_ix: number;
  metadata?: TetherQuoteMetadata;
  invalid_reason?: string;
}

interface PageState {
  aCounter: number;
  imgCounter: number;

  url: URL;
  url_string: string;
  hostname: string;
  links: string[];
  links_meta: TetherQuoteMetadata[];
  lines: string[];
  line_source: Record<string, Metadata>; // string representation of number interval
  title?: string;
}

interface BrowserState {
  pageStack: PageState[];
  quoteCounter: number;
  quotes: Record<string, Quote>;
}

function removeDenseLinks(document: Document, ratioThreshold: number = 0.5) {
  // Remove nav elements
  const navs = document.querySelectorAll('nav');
  navs.forEach(nav => {
    if (nav.parentNode) {
      nav.parentNode.removeChild(nav);
    }
  });

  // Query for lists, divs, spans, tables, and paragraphs
  const elements = document.querySelectorAll('ul, ol, div, span, nav, table, p');
  elements.forEach(element => {
    if (element === null) return;

    const children = Array.from(element.childNodes);
    const links = element.querySelectorAll('a');

    if (children.length <= 1) return;

    const allText = element.textContent ? element.textContent.trim().replace(/\s+/g, '') : '';
    const linksText = Array.from(links)
      .map(link => (link.textContent ? link.textContent.trim() : ''))
      .join('')
      .replace(/\s+/g, '');

    if (allText.length === 0 || linksText.length === 0) return;

    let ratio = linksText.length / allText.length;
    if (ratio > ratioThreshold && element.parentNode) {
      element.parentNode.removeChild(element);
    }
  });
}

abstract class BaseBrowser {
  public static toolName = 'browser' as const;
  public description = 'BaseBrowser';

  private turndownService = new TurndownService({
    headingStyle: 'atx',
  });

  private state: BrowserState;

  private transform(dom: JSDOM): string {
    let state = this.lastPageState();
    state.aCounter = 0;
    state.imgCounter = 0;
    state.links = [];

    return this.turndownService.turndown(dom.window.document);
  }

  private formatPage(state: PageState): string {
    let formatted_lines = state.lines.join('\n');
    let formatted_title = state.title ? `TITLE: ${state.title}\n\n` : '';
    let formatted_range = `\nVisible: 0% - 100%`;
    let formatted_message = formatted_title + formatted_lines + formatted_range;
    return formatted_message;
  }

  private newPageState(): PageState {
    return {
      aCounter: 0,
      imgCounter: 0,

      url: new URL('about:blank'),
      url_string: 'about:blank',
      hostname: '',
      title: '',
      links: [],
      links_meta: [],
      lines: [],
      line_source: {},
    };
  }

  private pushPageState(): PageState {
    let state = this.newPageState();
    this.state.pageStack.push(state);
    return state;
  }

  private lastPageState(): PageState {
    if (this.state.pageStack.length === 0) {
      throw new Error('No page state');
    }
    return this.state.pageStack[this.state.pageStack.length - 1];
  }

  private formatErrorUrl(url: string): string {
    let TRUNCATION_LIMIT = 80;
    if (url.length <= TRUNCATION_LIMIT) {
      return url;
    }
    return url.slice(0, TRUNCATION_LIMIT) + `... (URL truncated at ${TRUNCATION_LIMIT} chars)`;
  }

  protected functions = {
    search: async (query: string, recency_days: number = -1) => {
      logger.debug(`Searching for: ${query}`);
      const search = new URLSearchParams({ q: query });
      recency_days > 0 && search.append('recency_days', recency_days.toString());
      if (config.CUSTOM_CONFIG_ID) {
    search.append('customconfig', config.CUSTOM_CONFIG_ID.toString());
}
      const url = `${config.BING_SEARCH_API_URL}/search?${search.toString()}`;
      console.log('Full URL:', url); // 输出完整的 URL查看是否正确

      return withTimeout(
        config.BROWSER_TIMEOUT,
        fetch(url, {
          headers: {
            'Ocp-Apim-Subscription-Key': config.BING_SEARCH_API_KEY,
          }
        })
            .then(
          res =>
            res.json() as Promise<{
              queryContext: {
                originalQuery: string;
              };
              webPages: {
                webSearchUrl: string;
                totalEstimatedMatches: number;
                value: {
                  id: string;
                  name: string;
                  url: string;
                  datePublished: string; // 2018-05-18T08:00:00.0000000
                  datePublishedDisplayText: string;
                  isFamilyFriendly: boolean;
                  displayUrl: string;
                  snippet: string;
                  dateLastCrawled: string;
                  cachedPageUrl: string;
                  language: string;
                  isNavigational: boolean;
                }[];
              };
              rankingResponse: {
                mainline: {
                  items: {
                    answerType: 'WebPages';
                    resultIndex: number;
                    value: {
                      id: string;
                    };
                  }[];
                };
              };
            }>,
        ),
      )
        .then(async ({ value: res }) => {
          try {
            let state = this.pushPageState();
            let metadataList: TetherQuoteMetadata[] = [];
            for (const [i, entry] of res.webPages.value.entries()) {
              const url = new URL(entry.url);
              const hostname = url.hostname;
              state.lines.push(` # 【${i}†${entry.name}†${hostname}】`);
              state.lines.push(entry.snippet);
              const quoteMetadata: Metadata = {
                type: 'webpage',
                title: entry.name,
                url: entry.url,
                lines: state.lines.slice(2 * i, 2 * i + 2),
              };
              state.line_source[`${2 * i}-${2 * i + 1}`] = quoteMetadata;
              state.links[i] = entry.url;

              const returnMetadata: TetherQuoteMetadata = {
                type: quoteMetadata.type,
                title: quoteMetadata.title,
                url: quoteMetadata.url,
                text: state.lines[2 * i + 1], // only content, not link
                pub_date: entry.datePublished,
              };
              metadataList.push(returnMetadata);
            }
            const returnContentType = 'browser_result';
            return {
              contentType: returnContentType,
              roleMetadata: returnContentType,
              message: this.formatPage(state),
              metadataList,
            };
          } catch (err) {
            throw new Error(`parse error: ${err}`);
          }
        })
        .catch(err => {
          logger.error(`搜索请求失败：${query}，错误信息：${err.message}`);
          if (err.code === 'ECONNABORTED') {
            throw new Error(`Timeout while executing search for: ${query}`);
          }
          throw new Error(`网络或服务器发生错误，请检查URL: ${url}`);
        });
    },
    open_url: (url: string) => {
      logger.debug(`Opening ${url}`);

      return withTimeout(
        config.BROWSER_TIMEOUT,
        fetch(url).then(res => res.text()),
      )
        .then(async ({ value: res, time }) => {
          try {
            const state = this.pushPageState();
            state.url = new URL(url);
            state.url_string = url;
            state.hostname = state.url.hostname;

            const html = res;
            const dom = new JSDOM(html);
            const title = dom.window.document.title;
            const markdown = this.transform(dom);

            state.title = title;

            // Remove first line, because it will be served as the title
            const lines = markdown.split('\n');
            lines.shift();
            // Remove consequent empty lines
            let i = 0;
            while (i < lines.length - 1) {
              if (lines[i].trim() === '' && lines[i + 1].trim() === '') {
                lines.splice(i, 1);
              } else {
                i++;
              }
            }

            let page = lines.join('\n');

            // The first line feed is not a typo
            let text_result = `\nURL: ${url}\n${page}`;
            state.lines = text_result.split('\n');

            // all lines has only one source
            state.line_source = {};
            state.line_source[`0-${state.lines.length - 1}`] = {
              type: 'webpage',
              title: title,
              url: url,
              lines: state.lines,
            };

            let message = this.formatPage(state);

            const returnContentType = 'browser_result';
            return {
              contentType: returnContentType,
              roleMetadata: returnContentType,
              message,
              metadataList: state.links_meta,
            };
          } catch (err) {
            throw new Error(`parse error: ${err}`);
          }
        })
        .catch(err => {
          logger.error(err.message);
          if (err.code === 'ECONNABORTED') {
            throw new Error(`Timeout while loading page w/ URL: ${url}`);
          }
          throw new Error(`Failed to load page w/ URL: ${url}`);
        });
    },
    mclick: (ids: number[]) => {
      logger.info('Entering mclick', ids);
      let promises: Promise<ActionResult>[] = [];
      let state = this.lastPageState();
      for (let id of ids) {
        if (isNaN(id) || id >= state.links.length) {
          promises.push(
            Promise.reject(
              new Error(
                `recorded='click(${id})' temporary=None permanent=None new_state=None final=None success=False feedback='Error parsing ID ${id}' metadata={}`,
              ),
            ),
          );
          continue;
        }

        let url: string;
        try {
          url = new URL(state.links[id], state.url).href;
        } catch (err) {
          logger.error(`Failed in getting ${state.links[id]}, ${state.url}`);
          promises.push(
            Promise.reject(
              new Error(
                `recorded='click(${id})' temporary=None permanent='${err}' new_state=None final=None success=False feedback='Error parsing URL for ID ${id}' metadata={}`,
              ),
            ),
          );
          continue;
        }

        const quoteIndex = this.state.quoteCounter++; // ascending in final results
        promises.push(
          withTimeout(
            config.BROWSER_TIMEOUT,
            fetch(url).then(res => res.text()),
          )
            .then(({ value: res, time }) => {
              let state = this.newPageState();
              state.url = new URL(url);
              state.hostname = state.url.hostname;

              try {
                const html = res;
                const dom = new JSDOM(html);
                const title = dom.window.document.title;
                state.title = title;
                removeDenseLinks(dom.window.document);
                let quoteText = this.transform(dom);
                // remove consecutive newline
                quoteText = quoteText.replace(/[\r\n]+/g, '\n');
                const quoteLines = quoteText.split('\n');
                state.lines = quoteLines;
                const metadata = {
                  type: 'webpage',
                  title: title,
                  url: url,
                  lines: quoteLines,
                };
                const quoteMetadata = {
                  type: 'webpage',
                  title: title,
                  url: url,
                  text: quoteText,
                };
                state.line_source = {};
                state.line_source[`0-${state.lines.length - 1}`] = metadata;
                this.state.quotes[quoteIndex.toString()] = {
                  text: quoteText,
                  metadata: [metadata],
                };

                const returnContentType = 'quote_result';
                return {
                  contentType: returnContentType,
                  roleMetadata: `${returnContentType} [${quoteIndex}†source]`,
                  message: quoteText,
                  metadataList: [quoteMetadata],
                  metadata: {
                    url,
                  },
                };
              } catch (err) {
                throw new Error(`parse error: ${err}`);
              }
            })
            .catch(err => {
              logger.error(err.message);
              if (err.code === 'ECONNABORTED') {
                throw new Error(`Timeout while loading page w/ URL: ${this.formatErrorUrl(url)}`);
              }
              throw new Error(`Failed to load page w/ URL: ${this.formatErrorUrl(url)}`);
            })
            .catch(err => {
              // format error message
              const returnContentType = 'system_error';
              throw {
                contentType: returnContentType,
                roleMetadata: returnContentType,
                message: `recorded='click(${id})' temporary=None permanent='${
                  err.message
                }' new_state=None final=None success=False feedback='Error fetching url ${this.formatErrorUrl(
                  url,
                )}' metadata={}`,
                metadata: {
                  failedURL: url,
                },
              } as ActionResult;
            }),
        );
      }

      return Promise.allSettled(promises).then(async results => {
        const actionResults = results.map(r => {
          if (r.status === 'fulfilled') {
            return r.value;
          } else {
            logger.error(r.reason);
            return r.reason as ActionResult;
          }
        });

        if (results.filter(r => r.status === 'fulfilled').length === 0) {
          // collect errors
          const err_text = (results as PromiseRejectedResult[])
            .map(r => (r.reason as ActionResult).message)
            .join('\n');
          throw new Error(err_text);
        } else {
          return actionResults;
        }
      });
    },
  };

  constructor() {
    this.state =  {
      pageStack: [],
      quotes: {},
      quoteCounter: 7,
    };

    this.turndownService.remove('script');
    this.turndownService.remove('style');

    // Add rules for turndown
    this.turndownService.addRule('reference', {
      filter: function (node, options: any): boolean {
        return (
          options.linkStyle === 'inlined' &&
          node.nodeName === 'A' &&
          node.getAttribute('href') !== undefined
        );
      },

      replacement: (content, node, options): string => {
        let state = this.state.pageStack[this.state.pageStack.length - 1];
        if (!content || !('getAttribute' in node)) return '';
        let href = undefined;
        try {
          if ('getAttribute' in node) {
            const hostname = new URL(node.getAttribute('href')!).hostname;
            // Do not append hostname when in the same domain
            if (hostname === state.hostname || !hostname) {
              href = '';
            } else {
              href = '†' + hostname;
            }
          }
        } catch (e) {
          // To prevent displaying links like '/foo/bar'
          href = '';
        }
        if (href === undefined) return '';

        const url = node.getAttribute('href')!;
        let linkId = state.links.findIndex(link => link === url);
        if (linkId === -1) {
          linkId = state.aCounter++;
          // logger.debug(`New link[${linkId}]: ${url}`);
          state.links_meta.push({
            type: 'webpage',
            title: node.textContent!,
            url: href,
            text: node.textContent!,
          });
          state.links.push(url);
        }
        return `【${linkId}†${node.textContent}${href}】`;
      },
    });
    this.turndownService.addRule('img', {
      filter: 'img',

      replacement: (content, node, options): string => {
        let state = this.state.pageStack[this.state.pageStack.length - 1];
        return `[Image ${state.imgCounter++}]`;
      },
    });
    // Just to change indentation, wondering why this isn't exposed as an option
    this.turndownService.addRule('list', {
      filter: 'li',

      replacement: function (content, node, options) {
        content = content
          .replace(/^\n+/, '') // remove leading newlines
          .replace(/\n+$/, '\n') // replace trailing newlines with just a single one
          .replace(/\n/gm, '\n  '); // indent

        let prefix = options.bulletListMarker + ' ';
        const parent = node.parentNode! as Element;
        if (parent.nodeName === 'OL') {
          const start = parent.getAttribute('start');
          const index = Array.prototype.indexOf.call(parent.children, node);
          prefix = (start ? Number(start) + index : index + 1) + '.  ';
        }
        return '  ' + prefix + content + (node.nextSibling && !/\n$/.test(content) ? '\n' : '');
      },
    });
    // Remove bold; remove() doesn't work on this, I don't know why
    this.turndownService.addRule('emph', {
      filter: ['strong', 'b'],

      replacement: function (content, node, options) {
        if (!content.trim()) return '';
        return content;
      },
    });
  }

  abstract actionLine(content: string): Promise<ActionResult | ActionResult[]>;

  async action(content: string): Promise<ToolObservation[]> {
    const lines = content.split('\n');
    let results: ActionResult[] = [];
    for (const line of lines) {
      logger.info(`Action line: ${line}`)
      try {
        const lineActionResult = await this.actionLine(line);
        logger.debug(`Action line result: ${JSON.stringify(lineActionResult, null, 2)}`);
        if (Array.isArray(lineActionResult)) {
          results = results.concat(lineActionResult);
        } else {
          results.push(lineActionResult);
        }
      } catch (err) {
        const returnContentType = 'system_error';
        results.push({
          contentType: returnContentType,
          roleMetadata: returnContentType,
          message: `Error when executing command ${line}\n${err}`,
          metadata: {
            failedCommand: line,
          },
        });
      }
    }
    const observations: ToolObservation[] = [];
    for (const result of results) {
      const observation: ToolObservation = {
        contentType: result.contentType,
        result: result.message,
        roleMetadata: result.roleMetadata,
        metadata: result.metadata ?? {},
      };

      if (result.metadataList) {
        observation.metadata.metadata_list = result.metadataList;
      }
      observations.push(observation);
    }
    return observations;
  }

  postProcess(message: Message, metadata: any) {
    const quotePattern = /【(.+?)†(.*?)】/g;
    const content = message.content;
    let match;
    let citations: Citation[] = [];
    const citation_format_type = 'tether_og';
    while ((match = quotePattern.exec(content))) {
      logger.debug(`Citation match: ${match[0]}`);
      const start_ix = match.index;
      const end_ix = match.index + match[0].length;

      let invalid_reason = undefined;
      let metadata: TetherQuoteMetadata;
      try {
        let cited_message_idx = parseInt(match[1]);
        let evidence_text = match[2];
        let quote = this.state.quotes[cited_message_idx.toString()];
        if (quote === undefined) {
          invalid_reason = `'Referenced message ${cited_message_idx} in citation 【${cited_message_idx}†${evidence_text}】 is not a quote or tether browsing display.'`;
          logger.error(`Triggered citation error with quote undefined: ${invalid_reason}`);
          citations.push({
            citation_format_type,
            start_ix,
            end_ix,
            invalid_reason,
          });
        } else {
          let extra: TetherQuoteExtra = {
            cited_message_idx,
            evidence_text,
          };
          const quote_metadata = quote.metadata[0];
          metadata = {
            type: 'webpage',
            title: quote_metadata.title,
            url: quote_metadata.url,
            text: quote_metadata.lines.join('\n'),
            extra,
          };
          citations.push({
            citation_format_type,
            start_ix,
            end_ix,
            metadata,
          });
        }
      } catch (err) {
        logger.error(`Triggered citation error: ${err}`);
        invalid_reason = `Citation Error: ${err}`;
        citations.push({
          start_ix,
          end_ix,
          citation_format_type,
          invalid_reason,
        });
      }
    }
    metadata.citations = citations;
  }

  getState() {
    return this.state;
  }
}

export class SimpleBrowser extends BaseBrowser {
  public description = 'SimpleBrowser';

  constructor() {
    super();
  }

  async actionLine(content: string): Promise<ActionResult | ActionResult[]> {
    const regex = /(\w+)\(([^)]*)\)/;
    const matches = content.match(regex);

    if (matches) {
      const functionName = matches[1];
      let args_string = matches[2];
      if (functionName === 'mclick') {
        args_string = args_string.trim().slice(1, -1); // remove '[' and ']'
      }

      const args = args_string.split(',').map(arg => arg.trim());

      let result;
      switch (functionName) {
        case 'search':
          logger.debug(`SimpleBrowser action search ${args[0].slice(1, -1)}`);
          const recency_days = /(^|\D)(\d+)($|\D)/.exec(args[1])?.[2] as undefined | `${number}`;
          result = await this.functions.search(
            args[0].slice(1, -1), // slice quote "query"
            recency_days && Number(recency_days),
          );
          break;
        case 'open_url':
          logger.debug(`SimpleBrowser action open_url ${args[0].slice(1, -1)}`);
          result = await this.functions.open_url(args[0].slice(1, -1));
          break;
        case 'mclick':
          logger.debug(`SimpleBrowser action mclick ${args}`);
          result = await this.functions.mclick(args.map(x => parseInt(x)));
          break;
        default:
          throw new Error(`Parse Error: ${content}`);
      }

      return result;
    } else {
      throw new Error('Parse Error');
    }
  }
}

if (require.main === module) {
  (async () => {
    let browser = new SimpleBrowser();
    let demo = async (action: string) => {
      logger.info(` ------ Begin of Action: ${action} ------`);
      let results = await browser.action(action);
      for (const [idx, result] of results.entries()) {
        logger.info(`[Result ${idx}] contentType: ${result.contentType}`);
        logger.info(`[Result ${idx}] roleMetadata: ${result.roleMetadata}`);
        logger.info(`[Result ${idx}] result: ${result.result}`);
        logger.info(`[Result ${idx}] metadata: ${JSON.stringify(result.metadata, null, 2)}`);
      }
      logger.info(` ------ End of Action: ${action} ------\n\n`);
    };

    await demo("search('Apple Latest News')");
    await demo('mclick([0, 1, 5, 6])');
    await demo('mclick([1, 999999])');
    await demo("open_url('https://chatglm.cn')");
    await demo("search('zhipu latest News')");
    await demo('mclick([0, 1, 5, 6])');
  })();
}
