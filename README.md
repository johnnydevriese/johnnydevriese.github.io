# Johnny Devriese Personal Website

React + TypeScript + Vite blog with markdown source files in `content/posts`.

## Development

```bash
npm install
npm run dev
```

`npm run dev` regenerates `src/data/posts.ts` from `content/posts` before starting Vite.

## Adding Posts

1. Create a markdown file in `content/posts` using a date-prefixed filename:
   `YYYY-MM-DD-title-of-post.md`
2. Add frontmatter:

```yaml
---
title: "Your Post Title"
date: YYYY-MM-DD HH:MM:SS
categories: tag1 tag2 tag3
---
```

3. Run `npm run dev` or `npm run parse-posts`.

## Build

```bash
npm run build
```

## Content Pipeline

- `content/posts/` contains the source markdown posts
- `parse-posts.cjs` converts them into `src/data/posts.ts`
- `src/App.tsx` renders the generated post data
