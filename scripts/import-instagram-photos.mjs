import fs from 'node:fs/promises';
import path from 'node:path';

const sourceDir = process.argv[2];
const destinationDir = process.argv[3] ?? path.resolve('public/assets/images/photography/instagram');
const dataFile = process.argv[4] ?? path.resolve('src/data/photography.ts');
const postsHtmlFile = path.resolve(sourceDir, '../../your_instagram_activity/media/posts_1.html');
const excludedSourceFiles = new Set([
  '202506/18053954135586030.jpg',
  '202408/18069954802565390.jpg',
  '202312/17893116866870359.jpg',
]);

if (!sourceDir) {
  console.error('Usage: node scripts/import-instagram-photos.mjs <sourceDir> [destinationDir] [dataFile]');
  process.exit(1);
}

const monthFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
});

async function collectFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map(async (entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return collectFiles(fullPath);
    }
    return [fullPath];
  }));

  return files.flat();
}

function decodeHtmlEntities(value) {
  return value
    .replace(/&#039;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&#x27;/g, "'")
    .replace(/&#x2F;/g, '/')
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)));
}

function stripTags(value) {
  return decodeHtmlEntities(value.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim());
}

function formatCoordinate(value) {
  return Number(value).toFixed(4);
}

function buildLocation(metadata) {
  if (metadata?.latitude && metadata?.longitude) {
    return `${formatCoordinate(metadata.latitude)}, ${formatCoordinate(metadata.longitude)}`;
  }
  return 'Location Unavailable';
}

function buildTitle(metadata, index) {
  if (metadata?.caption) {
    return metadata.caption;
  }
  return `Archive Frame ${String(index + 1).padStart(3, '0')}`;
}

function buildDescription(metadata, dateLabel) {
  if (metadata?.caption) {
    return `${metadata.caption} · ${dateLabel}`;
  }
  return `Photo archive frame from ${dateLabel}.`;
}

function parsePostsMetadata(html) {
  const metadataByPath = new Map();
  const sections = html.split('<div class="pam _3-95 _2ph- _a6-g uiBoxWhite noborder">').slice(1);

  for (const section of sections) {
    const captionMatch = section.match(/<h2 class="_3-95 _2pim _a6-h _a6-i">([\s\S]*?)<\/h2>/);
    const dateMatch = section.match(/<div class="_3-94 _a6-o">([\s\S]*?)<\/div>/);
    const latitudeMatch = section.match(/<div class="_a6-q">Latitude<\/div><div><div class="_a6-q">([\s\S]*?)<\/div>/);
    const longitudeMatch = section.match(/<div class="_a6-q">Longitude<\/div><div><div class="_a6-q">([\s\S]*?)<\/div>/);
    const imageMatches = [...section.matchAll(/href="media\/posts\/([^"]+\.(?:jpg|jpeg|png|webp))"/gi)];

    const metadata = {
      caption: captionMatch ? stripTags(captionMatch[1]) : '',
      postedAt: dateMatch ? stripTags(dateMatch[1]) : '',
      latitude: latitudeMatch ? stripTags(latitudeMatch[1]) : '',
      longitude: longitudeMatch ? stripTags(longitudeMatch[1]) : '',
    };

    for (const match of imageMatches) {
      metadataByPath.set(match[1], metadata);
    }
  }

  return metadataByPath;
}

function toPhotoEntry(filePath, index, metadataByPath) {
  const monthFolder = path.basename(path.dirname(filePath));
  const fileName = path.basename(filePath);
  const [year, month] = [monthFolder.slice(0, 4), monthFolder.slice(4, 6)];
  const dateLabel = monthFormatter.format(new Date(`${year}-${month}-01T00:00:00.000Z`));
  const relativeFile = `${monthFolder}/${fileName}`;
  const metadata = metadataByPath.get(relativeFile);

  return {
    title: buildTitle(metadata, index),
    location: buildLocation(metadata),
    date: dateLabel,
    image: `/assets/images/photography/instagram/${monthFolder}-${fileName}`,
    description: buildDescription(metadata, dateLabel),
  };
}

const postsHtml = await fs.readFile(postsHtmlFile, 'utf8');
const metadataByPath = parsePostsMetadata(postsHtml);

const sourceFiles = (await collectFiles(sourceDir))
  .filter((file) => /\.(jpe?g|png|webp)$/i.test(file))
  .filter((file) => {
    const relativeFile = path.relative(sourceDir, file).split(path.sep).join('/');
    return !excludedSourceFiles.has(relativeFile);
  })
  .sort((left, right) => right.localeCompare(left));

await fs.mkdir(destinationDir, { recursive: true });

const existingDestinationFiles = await fs.readdir(destinationDir);
await Promise.all(existingDestinationFiles.map((file) => fs.rm(path.join(destinationDir, file), { force: true })));

await Promise.all(sourceFiles.map(async (sourceFile) => {
  const monthFolder = path.basename(path.dirname(sourceFile));
  const fileName = path.basename(sourceFile);
  const destinationFile = path.join(destinationDir, `${monthFolder}-${fileName}`);
  await fs.copyFile(sourceFile, destinationFile);
}));

const photography = sourceFiles.map((filePath, index) => toPhotoEntry(filePath, index, metadataByPath));

const fileContents = `// Auto-generated by scripts/import-instagram-photos.mjs
export interface Photo {
  title: string;
  location: string;
  date: string;
  image?: string;
  description: string;
}

export const photography: Photo[] = ${JSON.stringify(photography, null, 2)};\n`;

await fs.writeFile(dataFile, fileContents, 'utf8');

console.log(`Imported ${photography.length} photos from ${sourceDir}`);
console.log(`Copied assets into ${destinationDir}`);
console.log(`Wrote data file ${dataFile}`);
