import { marked, Renderer, type Tokens } from 'marked';
import hljs from 'highlight.js';

export function renderMarkdown(content: string, isDark: boolean, colors: any): string {
  const latexPlaceholders: { placeholder: string; original: string }[] = [];
  let idx = 0;

  function stashLatex(match: string): string {
    const ph = `LATEXPH${idx++}XEND`;
    latexPlaceholders.push({ placeholder: ph, original: match });
    return ph;
  }

  let src = content;

  // Preserve Liquid-style highlight blocks by converting to fenced code blocks
  src = src.replace(
    /\{%\s*highlight\s+(\w+)\s*%\}([\s\S]*?)\{%\s*endhighlight\s*%\}/g,
    (_m, lang, code) => `\`\`\`${lang}\n${code.trim()}\n\`\`\``
  );

  // Stash LaTeX (order matters: display before inline)
  src = src.replace(/\\\[([\s\S]*?)\\\]/g, (m) => stashLatex(m));
  src = src.replace(/\$\$([\s\S]*?)\$\$/g, (m) => stashLatex(m));
  src = src.replace(/\\\(([\s\S]*?)\\\)/g, (m) => stashLatex(m));
  src = src.replace(/\$([^\$\n]+?)\$/g, (m) => stashLatex(m));

  const font = "'IBM Plex Mono', monospace";
  const textColor = isDark ? colors.base.paper : colors.base.black;
  const linkColor = isDark ? colors.blue.light : colors.blue.DEFAULT;
  const codeBg = isDark ? colors.base[900] : colors.base[100];
  const borderColor = isDark ? colors.base[850] : colors.base[200];
  const blockquoteBorder = isDark ? colors.base[700] : colors.base[300];
  const blockquoteText = isDark ? colors.base[300] : colors.base[700];
  const hrColor = isDark ? colors.base[850] : colors.base[100];
  const imgBorder = isDark ? colors.base[850] : colors.base[100];

  // Custom renderer using regular functions so `this` binds to the Renderer instance,
  // giving access to this.parser.parseInline() / this.parser.parse() (marked v18 API).
  const renderer = new Renderer();

  renderer.heading = function (this: Renderer, { tokens, depth }: Tokens.Heading) {
    const text = this.parser.parseInline(tokens);
    const sizes: Record<number, string> = {
      1: '1.4rem', 2: '1.25rem', 3: '1.1rem',
      4: '1.05rem', 5: '1rem', 6: '0.95rem',
    };
    const margins = depth <= 2
      ? 'margin: 2rem 0 1rem 0;'
      : 'margin: 1.5rem 0 0.75rem 0;';
    return `<h${depth} style="font-family: ${font}; font-size: ${sizes[depth]}; font-weight: 600; ${margins}">${text}</h${depth}>`;
  };

  renderer.paragraph = function (this: Renderer, { tokens }: Tokens.Paragraph) {
    const text = this.parser.parseInline(tokens);
    if (text.startsWith('<img') || /^LATEXPH\d+XEND$/.test(text.trim())) {
      return text;
    }
    return `<p style="font-family: ${font}; font-size: 0.95rem; line-height: 1.7; margin-bottom: 1rem; color: ${textColor};">${text}</p>`;
  };

  renderer.code = function ({ text, lang }: Tokens.Code) {
    let highlighted: string;
    if (lang) {
      try {
        highlighted = hljs.highlight(text, { language: lang }).value;
      } catch {
        highlighted = escapeHtml(text);
      }
    } else {
      highlighted = escapeHtml(text);
    }
    const langClass = lang ? ` language-${lang}` : '';
    return `<pre style="background-color: ${codeBg}; padding: 1rem; border-radius: 4px; overflow-x: auto; margin: 1.5rem 0; border: 1px solid ${borderColor}; font-family: ${font}; font-size: 0.9rem; line-height: 1.6;"><code class="hljs${langClass}" style="font-family: inherit; display: block; white-space: pre;">${highlighted}</code></pre>`;
  };

  renderer.codespan = function ({ text }: Tokens.Codespan) {
    return `<code style="background-color: ${codeBg}; padding: 0.125rem 0.375rem; border-radius: 3px; font-family: ${font}; font-size: 0.9em;">${text}</code>`;
  };

  renderer.link = function (this: Renderer, { href, tokens }: Tokens.Link) {
    const text = this.parser.parseInline(tokens);
    return `<a href="${href}" target="_blank" rel="noopener noreferrer" style="color: ${linkColor}; text-decoration: underline;">${text}</a>`;
  };

  renderer.image = function ({ href, text, title }: Tokens.Image) {
    return `<img src="${href}" alt="${text}" title="${title || text}" style="max-width: 100%; height: auto; margin: 1.5rem 0; border-radius: 4px; border: 1px solid ${imgBorder};" />`;
  };

  renderer.blockquote = function (this: Renderer, { tokens }: Tokens.Blockquote) {
    const body = this.parser.parse(tokens);
    return `<blockquote style="border-left: 3px solid ${blockquoteBorder}; padding-left: 1rem; margin: 1rem 0; color: ${blockquoteText};">${body}</blockquote>`;
  };

  renderer.list = function (this: Renderer, token: Tokens.List) {
    const items = token.items.map((item) => this.listitem(item)).join('');
    const tag = token.ordered ? 'ol' : 'ul';
    const style = token.ordered ? 'list-style-type: decimal;' : 'list-style-type: disc;';
    return `<${tag} style="${style} margin: 1rem 0; padding-left: 1.5rem;">${items}</${tag}>`;
  };

  renderer.listitem = function (this: Renderer, item: Tokens.ListItem) {
    const body = this.parser.parse(item.tokens);
    return `<li style="margin-bottom: 0.5rem;">${body}</li>`;
  };

  renderer.hr = function () {
    return `<hr style="border: none; border-top: 1px solid ${hrColor}; margin: 2rem 0;" />`;
  };

  renderer.table = function (this: Renderer, token: Tokens.Table) {
    let headerHtml = '';
    for (const cell of token.header) {
      headerHtml += this.tablecell(cell);
    }
    headerHtml = `<tr style="border-bottom: 2px solid ${borderColor};">${headerHtml}</tr>`;

    let bodyHtml = '';
    for (const row of token.rows) {
      let rowHtml = '';
      for (const cell of row) {
        rowHtml += this.tablecell(cell);
      }
      bodyHtml += `<tr style="border-bottom: 1px solid ${borderColor};">${rowHtml}</tr>`;
    }

    return `<table style="width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-family: ${font}; font-size: 0.9rem;"><thead>${headerHtml}</thead><tbody>${bodyHtml}</tbody></table>`;
  };

  renderer.tablerow = function ({ text }: Tokens.TableRow) {
    return `<tr style="border-bottom: 1px solid ${borderColor};">${text}</tr>`;
  };

  renderer.tablecell = function (this: Renderer, token: Tokens.TableCell) {
    const text = this.parser.parseInline(token.tokens);
    const tag = token.header ? 'th' : 'td';
    const weight = token.header ? 'font-weight: 600;' : '';
    const alignStyle = token.align ? `text-align: ${token.align};` : 'text-align: left;';
    return `<${tag} style="padding: 0.5rem 0.75rem; ${alignStyle} ${weight}">${text}</${tag}>`;
  };

  renderer.strong = function (this: Renderer, { tokens }: Tokens.Strong) {
    return `<strong>${this.parser.parseInline(tokens)}</strong>`;
  };

  renderer.em = function (this: Renderer, { tokens }: Tokens.Em) {
    return `<em>${this.parser.parseInline(tokens)}</em>`;
  };

  let html = marked.parse(src, { renderer, async: false }) as string;

  // Restore LaTeX expressions
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
