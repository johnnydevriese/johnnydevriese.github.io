import { marked, Renderer, type Tokens } from 'marked';
import hljs from 'highlight.js';

function slugify(text: string): string {
  return text.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\w\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-').trim();
}

export function renderMarkdown(content: string): string {
  const latexPlaceholders: { placeholder: string; original: string }[] = [];
  let idx = 0;

  function stashLatex(match: string): string {
    const ph = `LATEXPH${idx++}XEND`;
    latexPlaceholders.push({ placeholder: ph, original: match });
    return ph;
  }

  let src = content;

  // Preserve Liquid-style highlight blocks
  src = src.replace(
    /\{%\s*highlight\s+(\w+)\s*%\}([\s\S]*?)\{%\s*endhighlight\s*%\}/g,
    (_m, lang, code) => `\`\`\`${lang}\n${code.trim()}\n\`\`\``
  );

  // Stash LaTeX (order matters: display before inline)
  src = src.replace(/\\\[([\s\S]*?)\\\]/g, (m) => stashLatex(m));
  src = src.replace(/\$\$([\s\S]*?)\$\$/g, (m) => stashLatex(m));
  src = src.replace(/\\\(([\s\S]*?)\\\)/g, (m) => stashLatex(m));
  src = src.replace(/\$([^\$\n]+?)\$/g, (m) => stashLatex(m));

  const renderer = new Renderer();

  renderer.heading = function (this: Renderer, { tokens, depth }: Tokens.Heading) {
    const text = this.parser.parseInline(tokens);
    const id = slugify(text);
    return `<h${depth} id="${id}" style="scroll-margin-top: 32px;">${text}</h${depth}>`;
  };

  renderer.paragraph = function (this: Renderer, { tokens }: Tokens.Paragraph) {
    const text = this.parser.parseInline(tokens);
    if (text.startsWith('<img') || /^LATEXPH\d+XEND$/.test(text.trim())) {
      return text;
    }
    return `<p>${text}</p>`;
  };

  renderer.code = function ({ text, lang }: Tokens.Code) {
    let highlighted: string;
    if (lang) {
      try { highlighted = hljs.highlight(text, { language: lang }).value; }
      catch { highlighted = escapeHtml(text); }
    } else {
      highlighted = escapeHtml(text);
    }
    const langClass = lang ? ` language-${lang}` : '';
    return `<pre><code class="hljs${langClass}">${highlighted}</code></pre>`;
  };

  renderer.codespan = function ({ text }: Tokens.Codespan) {
    return `<code>${text}</code>`;
  };

  renderer.link = function (this: Renderer, { href, tokens }: Tokens.Link) {
    const text = this.parser.parseInline(tokens);
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
  };

  renderer.image = function ({ href, text, title }: Tokens.Image) {
    return `<img src="${href}" alt="${text}" title="${title || text}" />`;
  };

  renderer.blockquote = function (this: Renderer, { tokens }: Tokens.Blockquote) {
    return `<blockquote>${this.parser.parse(tokens)}</blockquote>`;
  };

  renderer.list = function (this: Renderer, token: Tokens.List) {
    const items = token.items.map((item) => this.listitem(item)).join('');
    const tag = token.ordered ? 'ol' : 'ul';
    return `<${tag}>${items}</${tag}>`;
  };

  renderer.listitem = function (this: Renderer, item: Tokens.ListItem) {
    return `<li>${this.parser.parse(item.tokens)}</li>`;
  };

  renderer.hr = function () {
    return `<hr />`;
  };

  renderer.table = function (this: Renderer, token: Tokens.Table) {
    let headerHtml = '';
    for (const cell of token.header) headerHtml += this.tablecell(cell);
    headerHtml = `<tr>${headerHtml}</tr>`;

    let bodyHtml = '';
    for (const row of token.rows) {
      let rowHtml = '';
      for (const cell of row) rowHtml += this.tablecell(cell);
      bodyHtml += `<tr>${rowHtml}</tr>`;
    }

    return `<table><thead>${headerHtml}</thead><tbody>${bodyHtml}</tbody></table>`;
  };

  renderer.tablerow = function ({ text }: Tokens.TableRow) {
    return `<tr>${text}</tr>`;
  };

  renderer.tablecell = function (this: Renderer, token: Tokens.TableCell) {
    const text = this.parser.parseInline(token.tokens);
    const tag = token.header ? 'th' : 'td';
    const align = token.align ? ` style="text-align: ${token.align}"` : '';
    return `<${tag}${align}>${text}</${tag}>`;
  };

  renderer.strong = function (this: Renderer, { tokens }: Tokens.Strong) {
    return `<strong>${this.parser.parseInline(tokens)}</strong>`;
  };

  renderer.em = function (this: Renderer, { tokens }: Tokens.Em) {
    return `<em>${this.parser.parseInline(tokens)}</em>`;
  };

  let html = marked.parse(src, { renderer, async: false }) as string;

  for (const { placeholder, original } of latexPlaceholders) {
    html = html.replace(placeholder, () => original);
  }

  return html;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
