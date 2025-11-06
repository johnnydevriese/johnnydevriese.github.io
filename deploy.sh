#!/bin/bash

# Build the project
echo "Building project..."
npm run parse-posts
npm run build

# Deploy to GitHub Pages
echo "Deploying to GitHub Pages..."
git add dist -f
git commit -m "Deploy to GitHub Pages"
git subtree push --prefix dist origin gh-pages

echo "Deployment complete!"
