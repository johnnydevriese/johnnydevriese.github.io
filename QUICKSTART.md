# Quick Start Guide

## Your Blog Has Been Converted! 🎉

I've successfully converted your Jekyll blog to a modern React/TypeScript application with the following features:

### ✨ What's New

1. **Modern Design**: Beautiful Flexoki color scheme with dark/light mode toggle
2. **Individual Post Pages**: Click any post to view it in full detail
3. **Fast Performance**: Built with Vite for lightning-fast development and builds
4. **Type Safety**: Full TypeScript support
5. **Responsive**: Works beautifully on all devices
6. **MathJax Support**: Ready for mathematical notation

### 📁 Key Files Created

- `src/App.tsx` - Main blog component with routing logic
- `src/main.tsx` - React entry point
- `src/data/posts.ts` - Auto-generated from your `_posts` folder (26 posts converted!)
- `parse-posts.js` - Script to convert Jekyll posts to TypeScript
- `package.json` - Project dependencies
- `vite.config.ts` - Vite configuration
- `index.html` - HTML entry point
- `.github/workflows/deploy.yml` - Automatic deployment to GitHub Pages

### 🚀 Current Status

✅ Development server is running at http://localhost:5173
✅ All 26 blog posts have been converted
✅ Dark/light mode working
✅ Post navigation working
✅ Click any post to see the full content!

### 🎯 How to Use

#### View Your Blog
The dev server is already running! Open http://localhost:5173 in your browser.

#### Add New Posts
1. Create a new markdown file in `_posts` following the Jekyll convention:
   ```
   2025-11-05-my-new-post.md
   ```

2. Add frontmatter:
   ```yaml
   ---
   layout: post
   title: "My New Post"
   date: 2025-11-05 18:00:00
   categories: tag1 tag2 tag3
   ---
   ```

3. Regenerate the posts data:
   ```bash
   npm run parse-posts
   ```

#### Build for Production
```bash
npm run build
```

#### Deploy to GitHub Pages
The `.github/workflows/deploy.yml` file will automatically deploy your site when you push to the `master` branch.

Or manually:
```bash
chmod +x deploy.sh
./deploy.sh
```

### 🎨 Customization

#### Update Social Links
Edit `src/App.tsx` around line 363:
```typescript
const socials = [
  { label: 'Twitter', url: 'https://twitter.com/yourhandle' },
  { label: 'GitHub', url: 'https://github.com/yourusername' },
  { label: 'Email', url: 'mailto:your@email.com' }
];
```

#### Change Colors
The Flexoki color scheme is defined in `src/App.tsx`. Search for `const flexoki` to customize.

#### Update About Section
Edit the about content in `src/App.tsx` around line 349.

### 📦 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run parse-posts` - Convert Jekyll posts to TypeScript

### 🔧 Project Structure

```
johnnydevriese.github.io/
├── src/
│   ├── App.tsx              # Main blog component
│   ├── main.tsx             # React entry point
│   └── data/
│       └── posts.ts         # Generated posts (auto-generated)
├── _posts/                  # Your Jekyll posts (keep these!)
├── dist/                    # Build output (auto-generated)
├── parse-posts.js           # Post conversion script
├── package.json
├── vite.config.ts
├── tsconfig.json
└── index.html
```

### 🎓 Next Steps

1. **Customize**: Update social links, colors, and about section
2. **Test**: Click through all your posts to make sure they render correctly
3. **Deploy**: Push to GitHub and let the workflow deploy automatically
4. **Write**: Add new posts and watch them appear instantly!

### 💡 Tips

- The markdown renderer supports basic markdown syntax (headers, bold, italic, code, links, images)
- For advanced formatting, you can extend the `renderMarkdown` function in `App.tsx`
- MathJax is loaded automatically for mathematical notation
- The grain texture effect adds character but can be disabled by removing the `.grain` style

### 🐛 Troubleshooting

If posts don't appear:
```bash
npm run parse-posts
```

If styles look broken:
- Check that IBM Plex Mono font is loading
- Clear browser cache
- Restart dev server

Need to go back to Jekyll?
- Your original `_posts` are untouched
- All original files are preserved

---

Enjoy your new blog! 🚀
