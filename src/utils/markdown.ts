// Enhanced markdown renderer for better Jekyll post compatibility
import hljs from 'highlight.js';

export function renderMarkdown(content: string, isDark: boolean, colors: any): string {
  let html = content;
  
  // Preserve LaTeX expressions by replacing them with placeholders
  const latexExpressions: string[] = [];
  
  // Preserve display math \[...\]
  html = html.replace(/\\\[([\s\S]*?)\\\]/g, (_match, latex) => {
    const placeholder = `__LATEX_DISPLAY_${latexExpressions.length}__`;
    latexExpressions.push(`\\[${latex}\\]`);
    return placeholder;
  });
  
  // Preserve display math $$...$$
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, (_match, latex) => {
    const placeholder = `__LATEX_DISPLAY_${latexExpressions.length}__`;
    latexExpressions.push(`$$${latex}$$`);
    return placeholder;
  });
  
  // Preserve inline math $...$
  html = html.replace(/\$([^\$\n]+?)\$/g, (_match, latex) => {
    const placeholder = `__LATEX_INLINE_${latexExpressions.length}__`;
    latexExpressions.push(`$${latex}$`);
    return placeholder;
  });
  
  // Handle Jekyll code blocks with syntax highlighting
  html = html.replace(/\{%\s*highlight\s+(\w+)\s*%\}([\s\S]*?)\{%\s*endhighlight\s*%\}/g, 
    (_match, lang, code) => {
      const trimmedCode = code.trim();
      let highlighted;
      try {
        highlighted = hljs.highlight(trimmedCode, { language: lang }).value;
      } catch (e) {
        highlighted = escapeHtml(trimmedCode);
      }
      return `<pre style="background-color: ${isDark ? colors.base[900] : colors.base[100]}; padding: 1rem; border-radius: 4px; overflow-x: auto; margin: 1.5rem 0; border: 1px solid ${isDark ? colors.base[850] : colors.base[200]}; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; line-height: 1.6;"><code class="hljs language-${lang}" style="font-family: inherit; display: block; white-space: pre;">${highlighted}</code></pre>`;
  });
  
  // Handle Jekyll comments
  html = html.replace(/<!--[\s\S]*?-->/g, '');
  
  // Handle Jekyll links [text][ref]
  const linkRefs: {[key: string]: string} = {};
  html = html.replace(/\[([^\]]+)\]:\s*(.+)/g, (_match, ref, url) => {
    linkRefs[ref.toLowerCase()] = url.trim();
    return '';
  });
  
  // Replace reference-style links
  Object.keys(linkRefs).forEach(ref => {
    const regex = new RegExp(`\\[([^\\]]+)\\]\\[${ref}\\]`, 'gi');
    html = html.replace(regex, `<a href="${linkRefs[ref]}" target="_blank" rel="noopener noreferrer" style="color: ${isDark ? colors.blue.light : colors.blue.DEFAULT}; text-decoration: underline;">$1</a>`);
  });
  
  // Code blocks (triple backticks)
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_match, lang, code) => {
    const trimmedCode = code.trim();
    let highlighted;
    if (lang) {
      try {
        highlighted = hljs.highlight(trimmedCode, { language: lang }).value;
      } catch (e) {
        highlighted = escapeHtml(trimmedCode);
      }
    } else {
      highlighted = escapeHtml(trimmedCode);
    }
    const langClass = lang ? ` language-${lang}` : '';
    return `<pre style="background-color: ${isDark ? colors.base[900] : colors.base[100]}; padding: 1rem; border-radius: 4px; overflow-x: auto; margin: 1.5rem 0; border: 1px solid ${isDark ? colors.base[850] : colors.base[200]}; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; line-height: 1.6;"><code class="hljs${langClass}" style="font-family: inherit; display: block; white-space: pre;">${highlighted}</code></pre>`;
  });
  
  // Images with optional title
  html = html.replace(/!\[([^\]]*)\]\(([^\)]+?)(?:\s+"([^"]+)")?\)/g, (_match, alt, url, title) => {
    return `<img src="${url}" alt="${alt}" title="${title || alt}" style="max-width: 100%; height: auto; margin: 1.5rem 0; border-radius: 4px; border: 1px solid ${isDark ? colors.base[850] : colors.base[100]};" />`;
  });
  
  // Headers (must be before bold/italic)
  html = html.replace(/^######\s+(.+)$/gm, '<h6 style="font-family: IBM Plex Mono; font-size: 0.95rem; font-weight: 600; margin: 1.25rem 0 0.5rem 0;">$1</h6>');
  html = html.replace(/^#####\s+(.+)$/gm, '<h5 style="font-family: IBM Plex Mono; font-size: 1rem; font-weight: 600; margin: 1.25rem 0 0.5rem 0;">$1</h5>');
  html = html.replace(/^####\s+(.+)$/gm, '<h4 style="font-family: IBM Plex Mono; font-size: 1.05rem; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">$1</h4>');
  html = html.replace(/^###\s+(.+)$/gm, '<h3 style="font-family: IBM Plex Mono; font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">$1</h3>');
  html = html.replace(/^##\s+(.+)$/gm, '<h2 style="font-family: IBM Plex Mono; font-size: 1.25rem; font-weight: 600; margin: 2rem 0 1rem 0;">$1</h2>');
  html = html.replace(/^#\s+(.+)$/gm, '<h1 style="font-family: IBM Plex Mono; font-size: 1.4rem; font-weight: 600; margin: 2rem 0 1rem 0;">$1</h1>');
  
  // Bold and italic (order matters!)
  html = html.replace(/\*\*\*([^\*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/___([^_]+)___/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  html = html.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_]+)_/g, '<em>$1</em>');
  
  // Inline code (must be before links)
  html = html.replace(/`([^`]+)`/g, `<code style="background-color: ${isDark ? colors.base[900] : colors.base[100]}; padding: 0.125rem 0.375rem; border-radius: 3px; font-family: IBM Plex Mono; font-size: 0.9em;">$1</code>`);
  
  // Links (normal markdown links)
  html = html.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, `<a href="$2" target="_blank" rel="noopener noreferrer" style="color: ${isDark ? colors.blue.light : colors.blue.DEFAULT}; text-decoration: underline;">$1</a>`);
  
  // Horizontal rules
  html = html.replace(/^---$/gm, `<hr style="border: none; border-top: 1px solid ${isDark ? colors.base[850] : colors.base[100]}; margin: 2rem 0;" />`);
  html = html.replace(/^\*\*\*$/gm, `<hr style="border: none; border-top: 1px solid ${isDark ? colors.base[850] : colors.base[100]}; margin: 2rem 0;" />`);
  
  // Blockquotes
  html = html.replace(/^>\s+(.+)$/gm, `<blockquote style="border-left: 3px solid ${isDark ? colors.base[700] : colors.base[300]}; padding-left: 1rem; margin: 1rem 0; color: ${isDark ? colors.base[300] : colors.base[700]};">$1</blockquote>`);
  
  // Unordered lists
  html = html.replace(/^\*\s+(.+)$/gm, '<li style="margin-left: 1.5rem; margin-bottom: 0.5rem;">$1</li>');
  html = html.replace(/^-\s+(.+)$/gm, '<li style="margin-left: 1.5rem; margin-bottom: 0.5rem;">$1</li>');
  html = html.replace(/(<li[^>]*>.*<\/li>\n?)+/g, '<ul style="list-style-type: disc; margin: 1rem 0;">$&</ul>');
  
  // Ordered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li style="margin-left: 1.5rem; margin-bottom: 0.5rem;">$1</li>');
  // This is simplified - proper ordered lists would need more complex parsing
  
  // Line breaks
  html = html.replace(/  \n/g, '<br />');
  
  // Paragraphs
  html = html.split('\n\n').map(para => {
    para = para.trim();
    if (!para) return '';
    if (para.startsWith('<h') || 
        para.startsWith('<pre') || 
        para.startsWith('<img') || 
        para.startsWith('<hr') ||
        para.startsWith('<ul') ||
        para.startsWith('<ol') ||
        para.startsWith('<blockquote') ||
        para.startsWith('__LATEX_DISPLAY_')) {
      return para;
    }
    return `<p style="font-family: IBM Plex Mono; font-size: 0.95rem; line-height: 1.7; margin-bottom: 1rem; color: ${isDark ? colors.base.paper : colors.base.black};">${para}</p>`;
  }).join('\n');
  
  // Restore LaTeX expressions
  latexExpressions.forEach((latex, index) => {
    // We use a function as second argument to avoid special characters (like $)
    // in the replacement string being interpreted.
    const displayPlaceholder = `__LATEX_DISPLAY_${index}__`;
    const inlinePlaceholder = `__LATEX_INLINE_${index}__`;
    
    if (html.includes(displayPlaceholder)) {
      html = html.replace(displayPlaceholder, () => latex);
    }
    if (html.includes(inlinePlaceholder)) {
      html = html.replace(inlinePlaceholder, () => latex);
    }
  });
  
  return html;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
