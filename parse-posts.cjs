const fs = require("fs");
const path = require("path");

const postsDir = "./content/posts";
const outputFile = "./src/data/posts.ts";

function parseFrontMatter(content) {
  const frontMatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = content.match(frontMatterRegex);

  if (!match) {
    return { frontMatter: {}, content: content };
  }

  const [, frontMatterStr, contentStr] = match;
  const frontMatter = {};

  frontMatterStr.split("\n").forEach((line) => {
    const colonIndex = line.indexOf(":");
    if (colonIndex > 0) {
      const key = line.substring(0, colonIndex).trim();
      const value = line
        .substring(colonIndex + 1)
        .trim()
        .replace(/^["']|["']$/g, "");
      frontMatter[key] = value;
    }
  });

  return { frontMatter, content: contentStr.trim() };
}

function extractExcerpt(content, maxLength = 300) {
  // Remove markdown images
  let text = content.replace(/!\[.*?\]\(.*?\)/g, "");
  // Remove markdown links but keep text
  text = text.replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1");
  // Remove code blocks
  text = text.replace(/```[\s\S]*?```/g, "");
  // Remove inline code
  text = text.replace(/`[^`]+`/g, "");
  // Remove headers
  text = text.replace(/^#+\s+/gm, "");
  // Clean up whitespace
  text = text.replace(/\s+/g, " ").trim();

  if (text.length > maxLength) {
    text = text.substring(0, maxLength) + "...";
  }

  return text;
}

function generateSlug(filename) {
  // Remove date prefix and extension
  return filename
    .replace(/^\d{4}-\d{2}-\d{2}-/, "")
    .replace(/\.(md|markdown)$/, "");
}

function parseDate(dateStr, filename) {
  // Try to parse the date from frontmatter
  if (dateStr) {
    const date = new Date(dateStr);
    if (!isNaN(date.getTime())) {
      return date;
    }
  }

  // Fall back to filename
  const match = filename.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (match) {
    return new Date(`${match[1]}-${match[2]}-${match[3]}`);
  }

  return new Date();
}

function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  return `${year} · ${month}`;
}

function extractTags(categories) {
  if (!categories) return [];
  return categories.split(/\s+/).filter(Boolean).slice(0, 3);
}

const posts = [];

// Read all files in the content/posts directory
const files = fs
  .readdirSync(postsDir)
  .filter((f) => f.endsWith(".md") || f.endsWith(".markdown"));

files.forEach((filename) => {
  const filePath = path.join(postsDir, filename);
  const content = fs.readFileSync(filePath, "utf-8");
  const { frontMatter, content: markdownContent } = parseFrontMatter(content);

  const date = parseDate(frontMatter.date, filename);
  const slug = generateSlug(filename);

  posts.push({
    slug,
    date: formatDate(date),
    dateObj: date.toISOString(),
    title: frontMatter.title || "Untitled",
    excerpt: extractExcerpt(markdownContent),
    content: markdownContent,
    tags: extractTags(frontMatter.categories),
  });
});

// Sort by date (newest first)
posts.sort((a, b) => new Date(b.dateObj) - new Date(a.dateObj));

// Generate TypeScript file
const tsContent = `// Auto-generated from content/posts
// Run 'node parse-posts.cjs' to regenerate

export interface Post {
  slug: string;
  date: string;
  dateObj: string;
  title: string;
  excerpt: string;
  content: string;
  tags: string[];
}

export const posts: Post[] = ${JSON.stringify(posts, null, 2)};

export function getPostBySlug(slug: string): Post | undefined {
  return posts.find(post => post.slug === slug);
}

export function getAllPosts(): Post[] {
  return posts;
}
`;

// Create src/data directory if it doesn't exist
const dataDir = "./src/data";
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

fs.writeFileSync(outputFile, tsContent);

console.log(`✓ Parsed ${posts.length} posts`);
console.log(`✓ Generated ${outputFile}`);

// RSS Generation
const SITE_URL = "https://johnnydevriese.github.io";
const RSS_PATH = "./public/rss.xml";

function escapeXml(unsafe) {
  return unsafe.replace(/[<>&'"]/g, (c) => {
    switch (c) {
      case "<": return "&lt;";
      case ">": return "&gt;";
      case "&": return "&amp;";
      case "'": return "&apos;";
      case "\"": return "&quot;";
    }
  });
}

const rssItems = posts.map(post => `
    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${SITE_URL}/#${post.slug}</link>
      <guid>${SITE_URL}/#${post.slug}</guid>
      <pubDate>${new Date(post.dateObj).toUTCString()}</pubDate>
      <description>${escapeXml(post.excerpt)}</description>
    </item>`).join("");

const rssFeed = `<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Johnny Devriese</title>
    <link>${SITE_URL}</link>
    <description>Thoughts on programming, data science, and more.</description>
    <language>en-us</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${SITE_URL}/rss.xml" rel="self" type="application/rss+xml" />
    ${rssItems}
  </channel>
</rss>`;

// Ensure public directory exists
const publicDir = "./public";
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

fs.writeFileSync(RSS_PATH, rssFeed);
console.log(`✓ Generated ${RSS_PATH}`);
