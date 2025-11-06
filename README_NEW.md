# Johnny Devriese's Blog

A modern, minimalist blog built with React, TypeScript, and Vite, featuring a clean design with the Flexoki color scheme.

## Features

- 🎨 Beautiful Flexoki color scheme with dark/light mode
- 📝 Individual pages for each blog post
- 🔍 Click to read full articles
- 📱 Responsive design
- ⚡ Fast performance with Vite
- 🎯 Type-safe with TypeScript
- 🧮 MathJax support for mathematical notation

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Parse existing Jekyll posts (converts _posts to TypeScript data):
```bash
npm run parse-posts
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:5173`

## Development

### Adding New Posts

1. Add your markdown file to the `_posts` directory following the Jekyll naming convention:
   ```
   YYYY-MM-DD-title-of-post.md
   ```

2. Include frontmatter at the top:
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: YYYY-MM-DD HH:MM:SS
   categories: tag1 tag2 tag3
   ---
   ```

3. Run the parser to regenerate the posts data:
   ```bash
   npm run parse-posts
   ```

4. The new post will automatically appear in your blog

### Building for Production

```bash
npm run build
```

This will create an optimized build in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
├── src/
│   ├── App.tsx           # Main blog component with routing
│   ├── main.tsx          # React entry point
│   └── data/
│       └── posts.ts      # Generated posts data (auto-generated)
├── _posts/               # Jekyll-style markdown posts
├── parse-posts.js        # Script to convert posts to TypeScript
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Customization

### Colors

The blog uses the Flexoki color scheme defined in `App.tsx`. You can customize colors by modifying the `flexoki` object.

### Fonts

The blog uses IBM Plex Mono. Change the font in `index.html` and update the `fontFamily` styles in `App.tsx`.

### Social Links

Update the footer social links in `App.tsx`:

```typescript
const socials = [
  { label: 'Twitter', url: 'https://twitter.com/yourusername' },
  { label: 'GitHub', url: 'https://github.com/yourusername' },
  { label: 'Email', url: 'mailto:your@email.com' }
];
```

## Deployment

### GitHub Pages

1. Update `vite.config.ts` with your base path:
   ```typescript
   export default defineConfig({
     base: '/your-repo-name/',
     plugins: [react()],
   })
   ```

2. Build the project:
   ```bash
   npm run build
   ```

3. Deploy the `dist` directory to GitHub Pages

### Other Platforms

The built files in `dist` can be deployed to any static hosting service (Vercel, Netlify, etc.).

## License

MIT

## Credits

- Design inspired by modern minimalist principles
- Flexoki color scheme by Steph Ango
- Built with React, TypeScript, and Vite
