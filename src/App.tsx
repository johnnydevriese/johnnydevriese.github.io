import { useState, useEffect, useRef, useMemo } from 'react';
import { posts, Post } from './data/posts';
import { photography } from './data/photography';
import { renderMarkdown } from './utils/markdown';
import './styles.css';
import './highlight-theme.css';

declare global {
  interface Window {
    MathJax: any;
  }
}

type Section = 'posts' | 'about' | 'photography' | 'post-detail';

const archivedPostSlugs = new Set([
  'monet-cyclegan-tutorial',
  'exploring-features',
  'tpu-flowers',
]);

function decodeHtmlEntities(str: string): string {
  const el = document.createElement('textarea');
  el.innerHTML = str;
  return el.value;
}

function extractTocEntries(html: string): { id: string; text: string; level: number }[] {
  const entries: { id: string; text: string; level: number }[] = [];
  const regex = /<h([2-3])\s[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/gi;
  let match;
  while ((match = regex.exec(html)) !== null) {
    const text = decodeHtmlEntities(match[3].replace(/<[^>]+>/g, '').trim());
    entries.push({ id: match[2], text, level: parseInt(match[1]) });
  }
  return entries;
}

function getDisplayLocation(location: string): string | null {
  const trimmed = location.trim();
  if (!trimmed || trimmed === 'Location Unavailable') return null;
  if (/^-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?$/.test(trimmed)) return null;
  return trimmed;
}

export default function Blog() {
  const [activeSection, setActiveSection] = useState<Section>('posts');
  const [currentPost, setCurrentPost] = useState<Post | null>(null);
  const [selectedPhotoIndex, setSelectedPhotoIndex] = useState(0);
  const [activeTocId, setActiveTocId] = useState<string | null>(null);
  const photographyStripRef = useRef<HTMLDivElement>(null);
  const articleRef = useRef<HTMLElement>(null);

  // Auto-scroll photography strip
  useEffect(() => {
    if (activeSection === 'photography' && photographyStripRef.current) {
      const activeThumb = photographyStripRef.current.children[selectedPhotoIndex] as HTMLElement;
      if (activeThumb) {
        photographyStripRef.current.scrollTo({
          left: activeThumb.offsetLeft - (photographyStripRef.current.offsetWidth / 2) + (activeThumb.offsetWidth / 2),
          behavior: 'smooth'
        });
      }
    }
  }, [selectedPhotoIndex, activeSection]);

  // Browser history
  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      if (event.state) {
        setActiveSection(event.state.section);
        setCurrentPost(event.state.post || null);
      } else {
        const hash = window.location.hash.replace('#', '');
        const nextSection = hash === 'about' || hash === 'photography' ? hash : 'posts';
        setActiveSection(nextSection);
        setCurrentPost(null);
      }
    };
    window.addEventListener('popstate', handlePopState);
    if (!window.history.state) {
      const hash = window.location.hash.replace('#', '');
      const initialSection = hash === 'about' || hash === 'photography' ? hash : 'posts';
      window.history.replaceState({ section: initialSection, post: null }, '', initialSection === 'posts' ? '/' : `#${initialSection}`);
      setActiveSection(initialSection);
    }
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // MathJax
  useEffect(() => {
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']], processEscapes: true, processEnvironments: true },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
    };
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js';
    script.async = true;
    script.onload = () => { window.MathJax?.typesetPromise?.().catch(console.log); };
    document.head.appendChild(script);
    return () => { if (document.head.contains(script)) document.head.removeChild(script); };
  }, []);

  useEffect(() => {
    window.MathJax?.typesetPromise?.().catch(console.log);
  }, [activeSection, currentPost]);

  // Scroll-spy for TOC
  useEffect(() => {
    if (activeSection !== 'post-detail' || !articleRef.current) return;
    const headings = articleRef.current.querySelectorAll('h2[id], h3[id]');
    if (!headings.length) return;
    const visible = new Set<string>();
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) visible.add(e.target.id); else visible.delete(e.target.id);
      }
      for (const h of headings) {
        if (visible.has(h.id)) { setActiveTocId(h.id); return; }
      }
    }, { rootMargin: '-80px 0px -60% 0px', threshold: 0 });
    headings.forEach(h => obs.observe(h));
    return () => obs.disconnect();
  }, [activeSection, currentPost]);

  const navigateToPost = (post: Post) => {
    setCurrentPost(post);
    setActiveSection('post-detail');
    window.history.pushState({ section: 'post-detail', post }, '', `#${post.slug}`);
    window.scrollTo(0, 0);
  };

  const navigateBack = () => {
    setActiveSection('posts');
    setCurrentPost(null);
    window.history.pushState({ section: 'posts', post: null }, '', '/');
    window.scrollTo(0, 0);
  };

  const recentPosts = posts.filter(p => new Date(p.dateObj).getFullYear() >= 2021 && !archivedPostSlugs.has(p.slug));
  const archivePosts = posts.filter(p => new Date(p.dateObj).getFullYear() <= 2020 || archivedPostSlugs.has(p.slug));

  const renderedContent = useMemo(() => {
    if (!currentPost) return '';
    return renderMarkdown(currentPost.content);
  }, [currentPost]);

  const tocEntries = useMemo(() => extractTocEntries(renderedContent), [renderedContent]);

  const selectedPhoto = photography[selectedPhotoIndex] ?? photography[0];
  const photoCount = photography.length;
  const archiveStart = photography[photography.length - 1]?.date ?? '';
  const archiveEnd = photography[0]?.date ?? '';
  const archivedLocations = new Set(
    photography.map(p => getDisplayLocation(p.location)).filter((l): l is string => Boolean(l))
  ).size;
  const selectedPhotoLocation = getDisplayLocation(selectedPhoto.location);

  const isPostDetail = activeSection === 'post-detail';
  const isPhotography = activeSection === 'photography';

  const pageClass = isPostDetail && tocEntries.length > 0
    ? 'page page--has-toc'
    : isPhotography ? 'page page--wide' : 'page';

  return (
    <>
      <div className={pageClass}>
        {/* Header — spans full grid in TOC mode */}
        <header className="site-header" style={isPostDetail && tocEntries.length > 0 ? { gridColumn: '1 / -1' } : undefined}>
          <div className="site-header-inner">
            <img
              src="/johnnydevriese_profile_pic.jpg"
              alt="Johnny Devriese"
              className="profile-pic"
              onClick={() => isPostDetail && navigateBack()}
              style={{ cursor: isPostDetail ? 'pointer' : 'default' }}
            />
            <div className="site-header-text">
              <h1>
                <a href="/" onClick={(e) => { if (isPostDetail) { e.preventDefault(); navigateBack(); } }}>
                  Johnny Devriese
                </a>
              </h1>
              <p className="tagline">Thoughts on AI, research, and engineering</p>
            </div>
          </div>
          <nav>
            {!isPostDetail ? (
              ['posts', 'photography', 'about'].map(section => (
                <button
                  key={section}
                  className={activeSection === section ? 'active' : ''}
                  onClick={() => {
                    const s = section as Section;
                    setActiveSection(s);
                    window.history.pushState({ section: s, post: null }, '', s === 'posts' ? '/' : `#${section}`);
                  }}
                >
                  {section}
                </button>
              ))
            ) : (
              <button onClick={navigateBack}>← back to posts</button>
            )}
          </nav>
        </header>

        {/* TOC sidebar — only in post-detail with headings */}
        {isPostDetail && tocEntries.length > 0 && (
          <nav className="toc" aria-label="Contents">
            <div className="toc-label">Contents</div>
            <ul>
              {tocEntries.filter(e => e.level === 2).map(h2 => {
                const children = tocEntries.filter(e => e.level === 3 && tocEntries.indexOf(e) > tocEntries.indexOf(h2) && (tocEntries.findIndex((x, i) => i > tocEntries.indexOf(h2) && x.level === 2) === -1 || tocEntries.indexOf(e) < tocEntries.findIndex((x, i) => i > tocEntries.indexOf(h2) && x.level === 2)));
                return (
                  <li key={h2.id}>
                    <a href={`#${h2.id}`} className={activeTocId === h2.id ? 'active' : ''} onClick={e => { e.preventDefault(); document.getElementById(h2.id)?.scrollIntoView({ behavior: 'smooth' }); }}>{h2.text}</a>
                    {children.length > 0 && (
                      <ul>
                        {children.map(h3 => (
                          <li key={h3.id}>
                            <a href={`#${h3.id}`} className={activeTocId === h3.id ? 'active' : ''} onClick={e => { e.preventDefault(); document.getElementById(h3.id)?.scrollIntoView({ behavior: 'smooth' }); }}>{h3.text}</a>
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                );
              })}
            </ul>
          </nav>
        )}

        {/* Main content area */}
        <main className="page-main">
          {/* Posts listing */}
          {activeSection === 'posts' && (
            <div>
              <div className="post-list">
                {recentPosts.map(post => (
                  <article key={post.slug} className="post-list-item" onClick={() => navigateToPost(post)} style={{ cursor: 'pointer' }}>
                    <div className="post-list-date">{post.date}</div>
                    <h2 className="post-list-title">
                      <a href={`#${post.slug}`} onClick={e => e.preventDefault()}>{post.title}</a>
                    </h2>
                    <p className="post-list-excerpt">{post.excerpt}</p>
                    <div className="post-list-tags">
                      {post.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
                    </div>
                  </article>
                ))}
              </div>

              {archivePosts.length > 0 && (
                <div style={{ marginTop: '48px', paddingTop: '32px', borderTop: '2px solid var(--rule)' }}>
                  <h2 style={{ fontFamily: "'Source Serif 4', serif", fontSize: '1.4rem', fontWeight: 600, marginBottom: '20px' }}>Archive</h2>
                  <ul className="archive-list">
                    {archivePosts.map(post => (
                      <li key={post.slug} className="archive-item" onClick={() => navigateToPost(post)} style={{ cursor: 'pointer' }}>
                        <span className="archive-date">{post.date}</span>
                        <span className="archive-title"><a href={`#${post.slug}`} onClick={e => e.preventDefault()}>{post.title}</a></span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Post detail */}
          {isPostDetail && currentPost && (
            <article ref={articleRef} style={{ padding: '40px 0 80px' }}>
              <div className="post-eyebrow">{currentPost.date}</div>
              <h1 className="post-title">{currentPost.title}</h1>
              <div
                className="post-content"
                dangerouslySetInnerHTML={{ __html: renderedContent }}
              />
              <div className="post-tags">
                {currentPost.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
              </div>
            </article>
          )}

          {/* About */}
          {activeSection === 'about' && (
            <div className="about">
              <p>
                I build production GenAI systems. Background in physics from WSU, graduate AI coursework from Stanford. Most of my work is figuring out how to take interesting research and turn it into reliable systems — currently focused on RAG pipelines, LLM evaluation, and deploying models that actually work at scale. I write about the engineering decisions that don't fit in academic papers.
              </p>
            </div>
          )}

          {/* Photography */}
          {isPhotography && (
            <section className="photography-shell">
              <div className="photography-rail">
                <p className="photography-kicker" style={{ margin: 0 }}>Photography Archive</p>
                <p className="photography-body" style={{ margin: 0, fontSize: '0.78rem' }}>
                  {archiveStart} → {archiveEnd} · {photoCount} frames · {archivedLocations} places
                </p>
              </div>

              <div className="photography-stage">
                <div className="photography-frame-panel">
                  <div className="photography-frame">
                    {selectedPhoto.image ? (
                      <img src={selectedPhoto.image} alt={selectedPhoto.title} />
                    ) : (
                      <span className="photography-kicker">Add Photo</span>
                    )}
                  </div>
                  <div className="photography-frame-rail">
                    <p className="photography-kicker" style={{ margin: 0 }}>Frame {String(selectedPhotoIndex + 1).padStart(2, '0')}</p>
                    <p className="photography-body" style={{ margin: 0, fontSize: '0.78rem' }}>
                      {selectedPhotoLocation ? `${selectedPhotoLocation} · ${selectedPhoto.date}` : selectedPhoto.date}
                    </p>
                  </div>
                </div>

                <div className="photography-meta-panel">
                  <div className="photography-meta-block">
                    <h3 className="photography-stage-title">{selectedPhoto.title}</h3>
                    <div className="photography-detail-list">
                      {selectedPhotoLocation && (
                        <p className="photography-kicker" style={{ margin: 0 }}>{selectedPhotoLocation}</p>
                      )}
                      <p className="photography-kicker" style={{ margin: 0 }}>Captured {selectedPhoto.date}</p>
                    </div>
                    <p className="photography-body" style={{ margin: 0 }}>{selectedPhoto.description}</p>
                  </div>

                  <div className="photography-actions">
                    <button className="photography-action" onClick={() => setSelectedPhotoIndex((selectedPhotoIndex - 1 + photography.length) % photography.length)}>
                      Prev
                    </button>
                    <button className="photography-action is-primary" onClick={() => setSelectedPhotoIndex((selectedPhotoIndex + 1) % photography.length)}>
                      Next
                    </button>
                  </div>
                </div>
              </div>

              <div className="photography-strip-container">
                <p className="photography-kicker" style={{ marginTop: 0 }}>Browse the Archive</p>
                <div className="photography-strip" ref={photographyStripRef}>
                  {photography.map((photo, index) => (
                    <button
                      key={`${photo.title}-${photo.date}-${index}`}
                      className={`photography-thumb${selectedPhotoIndex === index ? ' is-active' : ''}`}
                      onClick={() => setSelectedPhotoIndex(index)}
                    >
                      {photo.image ? (
                        <img
                          className="photography-thumb-media"
                          src={photo.image}
                          alt={photo.title}
                          style={{ filter: selectedPhotoIndex === index ? 'none' : 'grayscale(0.4) contrast(0.9)' }}
                        />
                      ) : (
                        <div className="photography-thumb-media" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <span className="photography-kicker">{String(index + 1).padStart(2, '0')}</span>
                        </div>
                      )}
                      <div className="photography-thumb-copy">
                        {photo.title}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          )}
        </main>

        {/* Footer */}
        <footer className="site-footer" style={isPostDetail && tocEntries.length > 0 ? { gridColumn: '1 / -1' } : undefined}>
          <div>
            <p>© 2026 Johnny Devriese</p>
            <p style={{ marginTop: '2px' }}>human made</p>
            <img
              className="footer-badge"
              src="/assets/images/footer/kagifeedback-88x31.gif"
              alt="Fun feedback badge"
              width="88"
              height="31"
              loading="lazy"
            />
          </div>
          <div className="social-icons">
            <a href="https://github.com/johnnydevriese/" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405 1.02 0 2.04.135 3 .405 2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.285 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
            </a>
            <a href="https://bsky.app/profile/johnnydevriese.bsky.social" target="_blank" rel="noopener noreferrer" aria-label="Bluesky">
              <svg viewBox="0 0 24 24"><path d="M12 10.8c-1.087-2.114-4.046-6.053-6.798-7.995C2.566.944 1.561 1.266.902 1.565.139 1.908 0 3.08 0 3.768c0 .69.378 5.65.624 6.479.815 2.736 3.713 3.66 6.383 3.364.136-.02.275-.039.415-.056-.138.022-.276.04-.415.056-3.912.58-7.387 2.005-2.83 7.078 5.013 5.19 6.617 2 7.823 2 1.205 0 2.81 3.19 7.822-2 4.558-5.073 1.083-6.498-2.829-7.078-.139-.016-.277-.034-.415-.056.14.017.279.036.415.056 2.67.297 5.568-.628 6.383-3.364.246-.828.624-5.79.624-6.478 0-.69-.139-1.861-.902-2.206-.659-.298-1.664-.62-4.3 1.24C16.046 4.748 13.087 8.687 12 10.8z"/></svg>
            </a>
            <a href="https://www.linkedin.com/in/johnny-devriese-080556129/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
              <svg viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
            </a>
            <a href="https://x.com/johnnydevriese" target="_blank" rel="noopener noreferrer" aria-label="X">
              <svg viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
            </a>
          </div>
        </footer>
      </div>
    </>
  );
}
