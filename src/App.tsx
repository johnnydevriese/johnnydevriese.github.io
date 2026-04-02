import { useState, useEffect, useRef } from 'react';
import { posts, Post } from './data/posts';
import { photography } from './data/photography';
import { renderMarkdown } from './utils/markdown';
import { SocialIcons } from './components/SocialIcons';
import './highlight-theme.css';

declare global {
  interface Window {
    MathJax: any;
  }
}

type Section = 'posts' | 'about' | 'photography' | 'post-detail';

export default function Blog() {
  const [isDark, setIsDark] = useState(() => {
    // Check system preference on initial load
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [activeSection, setActiveSection] = useState<Section>('posts');
  const [currentPost, setCurrentPost] = useState<Post | null>(null);
  const [selectedPhotoIndex, setSelectedPhotoIndex] = useState(0);
  const photographyStripRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logic for photography strip
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

  // Handle browser back/forward buttons
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
    
    // Set initial state
    if (!window.history.state) {
      const hash = window.location.hash.replace('#', '');
      const initialSection = hash === 'about' || hash === 'photography' ? hash : 'posts';
      window.history.replaceState(
        { section: initialSection, post: null },
        '',
        initialSection === 'posts' ? '/' : `#${initialSection}`
      );
      setActiveSection(initialSection);
    }

    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  // Update data-theme attribute when theme changes
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      setIsDark(e.matches);
    };
    
    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
    // Older browsers
    else if (mediaQuery.addListener) {
      mediaQuery.addListener(handleChange);
      return () => mediaQuery.removeListener(handleChange);
    }
  }, []);

  useEffect(() => {
    // MathJax configuration
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };

    // Load MathJax
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js';
    script.async = true;
    script.onload = () => {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise().catch((err: any) => console.log(err));
      }
    };
    document.head.appendChild(script);

    return () => {
      if (document.head.contains(script)) {
        document.head.removeChild(script);
      }
    };
  }, []);

  useEffect(() => {
    // Rerender MathJax when content changes
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise().catch((err: any) => console.log(err));
    }
  }, [activeSection, isDark, currentPost]);

  // Kanagawa Theme Colors
  const kanagawa = {
    // Dragon (Dark)
    dragon: {
      bg: '#181616',        // dragonBlack0
      bgSoft: '#1D1C19',    // dragonBlack1
      bgSofter: '#282727',  // dragonBlack3
      fg: '#C5C9C5',        // dragonWhite
      fgSoft: '#A6A69C',    // dragonGray
      fgDim: '#737C73',     // dragonGray3
      red: '#C4746E',       // dragonRed
      orange: '#B6927B',    // dragonOrange
      yellow: '#C4B28A',    // dragonYellow
      green: '#87A987',     // dragonGreen
      cyan: '#8EA4A2',      // dragonAqua
      blue: '#8BA4B0',      // dragonBlue
      purple: '#A292A3',    // dragonViolet
      pink: '#B98D7B',      // dragonPink
    },
    // Lotus (Light)
    lotus: {
      bg: '#F2F4F8',        // lotusWhite0
      bgSoft: '#E6ECF1',    // lotusWhite2
      bgSofter: '#DCDFE4',  // lotusWhite4
      fg: '#43444B',        // lotusInk1
      fgSoft: '#545464',    // lotusInk2
      fgDim: '#716E61',     // lotusGray
      red: '#C84053',       // lotusRed
      orange: '#CC6D00',    // lotusOrange
      yellow: '#77713F',    // lotusYellow
      green: '#6F894E',     // lotusGreen
      cyan: '#4D696B',      // lotusAqua
      blue: '#4E618D',      // lotusBlue
      purple: '#9F7186',    // lotusViolet
      pink: '#B35B79',      // lotusPink
    }
  };

  const lightBg = kanagawa.lotus.bg;
  const lightText = kanagawa.lotus.fg;
  const lightBorder = kanagawa.lotus.bgSofter;
  const lightMuted = kanagawa.lotus.fgSoft;
  const lightDimmed = kanagawa.lotus.fgDim;

  const darkBg = kanagawa.dragon.bg;
  const darkText = kanagawa.dragon.fg;
  const darkBorder = kanagawa.dragon.bgSofter;
  const darkMuted = kanagawa.dragon.fgSoft;
  const darkDimmed = kanagawa.dragon.fgDim;

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

  // Separate posts into recent and archive based on year
  const recentPosts = posts.filter(post => {
    const year = new Date(post.dateObj).getFullYear();
    return year >= 2021;
  });

  const archivePosts = posts.filter(post => {
    const year = new Date(post.dateObj).getFullYear();
    return year <= 2020;
  });

  const renderPostContent = (content: string) => {
    // We pass the kanagawa theme colors to the markdown renderer
    // Creating a compatible object for the markdown renderer
    // Note: 'paper' is used for text color in dark mode, 'black' for light mode
    const themeColors = {
      base: {
        paper: isDark ? darkText : lightBg,  // text color in dark mode
        black: isDark ? darkBg : lightText,  // text color in light mode
        100: isDark ? darkBorder : lightBorder,
        200: isDark ? kanagawa.dragon.bgSofter : kanagawa.lotus.bgSofter,
        300: isDark ? darkMuted : lightMuted,
        700: isDark ? darkDimmed : lightDimmed,
        850: isDark ? darkBorder : lightBorder,
        900: isDark ? kanagawa.dragon.bgSoft : kanagawa.lotus.bgSoft,
      },
      orange: { 
        DEFAULT: isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange, 
        light: isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange 
      },
      blue: {
        DEFAULT: isDark ? kanagawa.dragon.blue : kanagawa.lotus.blue,
        light: isDark ? kanagawa.dragon.blue : kanagawa.lotus.blue
      }
    };
    
    return renderMarkdown(content, isDark, themeColors);
  };

  const getDisplayLocation = (location: string) => {
    const trimmed = location.trim();
    if (!trimmed || trimmed === 'Location Unavailable') {
      return null;
    }

    const coordinatePattern = /^-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?$/;
    if (coordinatePattern.test(trimmed)) {
      return null;
    }

    return trimmed;
  };

  const selectedPhoto = photography[selectedPhotoIndex] ?? photography[0];
  const photoCount = photography.length;
  const archiveStart = photography[photography.length - 1]?.date ?? '';
  const archiveEnd = photography[0]?.date ?? '';
  const archivedLocations = new Set(
    photography
      .map((photo) => getDisplayLocation(photo.location))
      .filter((location): location is string => Boolean(location))
  ).size;
  const selectedPhotoLocation = getDisplayLocation(selectedPhoto.location);
  const photographyAccent = isDark ? '#E7DED0' : '#1F2024';
  const photographyAccentAlt = isDark ? '#9CA3AF' : '#6B7280';
  const photographyFrameBorder = isDark ? '#D7D3CB' : '#23252B';
  const photographyShadow = `4px 4px 0px ${photographyFrameBorder}`;
  const photographyDisplayFont = "'Instrument Serif', serif";

  return (
    <div>
      <style>{`
        @keyframes grain {
          0% { transform: translate(0, 0); }
          10% { transform: translate(-5px, -5px); }
          20% { transform: translate(-10px, 5px); }
          30% { transform: translate(5px, -10px); }
          40% { transform: translate(-5px, 5px); }
          50% { transform: translate(-10px, -5px); }
          60% { transform: translate(5px, 5px); }
          70% { transform: translate(-5px, -10px); }
          80% { transform: translate(10px, 5px); }
          90% { transform: translate(-10px, 10px); }
          100% { transform: translate(0, 0); }
        }
        .grain {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' seed='2' /%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noiseFilter)' opacity='0.05'/%3E%3C/svg%3E");
          animation: grain 8s steps(10) infinite;
          z-index: 0;
        }
        .content-wrapper {
          position: relative;
          z-index: 1;
        }
        .math-content {
          overflow-x: auto;
        }
        .photography-shell {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          margin-top: 0;
        }
        .photography-rail {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 1rem;
          flex-wrap: wrap;
          padding-bottom: 0.2rem;
        }
        .photography-stage {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        .photography-frame-panel {
          background: transparent;
          padding: 0;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .photography-frame {
          position: relative;
          aspect-ratio: 16 / 9;
          overflow: hidden;
          border: 1px solid ${isDark ? '#3B3A38' : '#CFC5B7'};
          background: ${isDark ? '#141414' : '#EEE7DB'};
        }
        .photography-frame-rail {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 1rem;
          padding: 0;
          flex-wrap: wrap;
        }
        .photography-meta-panel {
          border-top: 1px solid ${isDark ? '#343434' : '#D8D0C4'};
          background: transparent;
          box-shadow: none;
          padding: 1rem 0 0 0;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 1.5rem;
        }
        .photography-strip-container {
          position: relative;
          margin-top: 0.5rem;
          border-top: 1px solid ${isDark ? '#343434' : '#D8D0C4'};
          padding-top: 1rem;
        }
        .photography-strip {
          display: flex;
          gap: 1rem;
          overflow-x: auto;
          scroll-snap-type: x mandatory;
          padding: 0.25rem 0 1.5rem 0;
          -ms-overflow-style: none;  /* IE and Edge */
          scrollbar-width: none;  /* Firefox */
        }
        .photography-strip::-webkit-scrollbar {
          display: none; /* Chrome, Safari and Opera */
        }
        .photography-thumb {
          flex: 0 0 260px;
          scroll-snap-align: start;
          border: 2px solid ${photographyFrameBorder};
          background: ${isDark ? '#161616' : '#FDFBF7'};
          box-shadow: ${photographyShadow};
          overflow: hidden;
          text-align: left;
          padding: 0;
          cursor: pointer;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          display: flex;
          flex-direction: column;
          align-self: flex-start;
        }
        .photography-thumb:hover {
          transform: translate(-4px, -4px);
          box-shadow: 8px 8px 0px ${photographyFrameBorder};
        }
        .photography-thumb.is-active {
          background: ${isDark ? '#222222' : '#F1EBE0'};
          transform: translate(-2px, -2px);
          box-shadow: 6px 6px 0px ${photographyFrameBorder};
        }
        .photography-thumb-media {
          aspect-ratio: 16 / 9;
          border-bottom: 2px solid ${photographyFrameBorder};
          overflow: hidden;
          background: ${isDark ? '#1A1A1A' : '#EFE9DE'};
        }
        .photography-thumb-copy {
          padding: 0.8rem 0.95rem 0.95rem 0.95rem;
        }
        .photography-kicker {
          font-family: IBM Plex Mono;
          font-size: 0.62rem;
          font-weight: 600;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: ${photographyAccentAlt};
          margin-bottom: 0.45rem;
        }
        .photography-title {
          font-family: ${photographyDisplayFont};
          font-size: clamp(1.45rem, 2.9vw, 2.35rem);
          line-height: 0.94;
          letter-spacing: -0.04em;
          color: ${photographyAccent};
          margin: 0;
        }
        .photography-stage-title {
          font-family: ${photographyDisplayFont};
          font-size: clamp(1.6rem, 2.4vw, 2.2rem);
          line-height: 0.98;
          letter-spacing: -0.03em;
          margin: 0;
          color: ${photographyAccent};
        }
        .photography-body {
          font-family: IBM Plex Mono;
          font-size: 0.78rem;
          line-height: 1.6;
          color: ${isDark ? darkText : lightText};
        }
        .photography-meta-block {
          display: flex;
          flex-direction: column;
          gap: 0.8rem;
          max-width: 44rem;
        }
        .photography-detail-list {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
        }
        .photography-actions {
          display: flex;
          gap: 0.9rem;
          flex-shrink: 0;
        }
        .photography-action {
          border: 1px solid ${isDark ? '#4A4946' : '#BFB5A6'};
          background: transparent;
          color: ${isDark ? '#F5F1E8' : '#23252B'};
          padding: 0.7rem 1rem;
          font-family: IBM Plex Mono;
          font-size: 0.68rem;
          font-weight: 600;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          cursor: pointer;
          box-shadow: none;
          transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
        }
        .photography-action:hover {
          background: ${isDark ? '#1A1A1A' : '#F4EEE4'};
        }
        .photography-action:active {
          background: ${isDark ? '#202020' : '#ECE3D6'};
        }
        .photography-action.is-primary {
          background: ${isDark ? '#EFE5D5' : '#23252B'};
          color: ${isDark ? '#111111' : '#FDFBF7'};
          border-color: ${isDark ? '#EFE5D5' : '#23252B'};
        }
        @media (max-width: 960px) {
          .photography-meta-panel {
            flex-direction: column;
            align-items: stretch;
          }
          .photography-thumb {
            flex-basis: 220px;
          }
        }
        @media (max-width: 640px) {
          .photography-actions {
            width: 100%;
          }
          .photography-action {
            flex: 1;
          }
        }
        /* Selection color */
        ::selection {
          background: ${isDark ? kanagawa.dragon.bgSofter : kanagawa.lotus.bgSofter};
          color: ${isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange};
        }
      `}</style>
      <div style={{
        backgroundColor: isDark ? darkBg : lightBg,
        color: isDark ? darkText : lightText,
        minHeight: '100vh',
        transition: 'background-color 0.3s, color 0.3s'
      }}>
        <div className="grain" style={{filter: isDark ? 'invert(1) brightness(1.1)' : 'invert(0)'}}></div>
        <div className="content-wrapper" style={{
          maxWidth: activeSection === 'photography' ? '72rem' : '42rem',
          margin: '0 auto',
          padding: activeSection === 'photography' ? '1rem 2rem 4rem 2rem' : '4rem 2rem',
          transition: 'max-width 0.3s ease, padding 0.3s ease'
        }}>
          
          {/* Header */}
          <header style={{
            marginBottom: activeSection === 'photography' ? '1.25rem' : '3rem',
            borderBottomColor: isDark ? darkBorder : lightBorder,
            borderBottomWidth: '1px',
            paddingBottom: activeSection === 'photography' ? '1.25rem' : '2.5rem'
          }}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 0, flexWrap: 'wrap', gap: activeSection === 'photography' ? '0.75rem' : '1rem'}}>
              <div style={{display: 'flex', gap: activeSection === 'photography' ? '0.85rem' : '1rem', alignItems: 'flex-start'}}>
                <div style={{
                  width: activeSection === 'photography' ? '56px' : '64px',
                  height: activeSection === 'photography' ? '56px' : '64px',
                  minWidth: activeSection === 'photography' ? '56px' : '64px',
                  borderRadius: '50%',
                  backgroundColor: isDark ? kanagawa.dragon.bgSoft : kanagawa.lotus.bgSoft,
                  border: `2px solid ${isDark ? darkBorder : lightBorder}`,
                  backgroundImage: 'url(/johnnydevriese_profile_pic.jpg)',
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  cursor: activeSection === 'post-detail' ? 'pointer' : 'default'
                }}
                onClick={() => activeSection === 'post-detail' && navigateBack()}
                ></div>
                <div>
                  <h1 
                    style={{
                      fontSize: activeSection === 'photography' ? '1.35rem' : '1.5rem', 
                      fontWeight: 700, 
                      marginBottom: activeSection === 'photography' ? '0.2rem' : '0.375rem', 
                      fontFamily: 'IBM Plex Mono',
                      cursor: activeSection === 'post-detail' ? 'pointer' : 'default',
                      color: isDark ? '#3AA99F' : '#24837B'
                    }}
                    onClick={() => activeSection === 'post-detail' && navigateBack()}
                  >
                    Johnny Devriese
                  </h1>
                  <p style={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: activeSection === 'photography' ? '0.82rem' : '0.875rem',
                    color: isDark ? darkMuted : lightMuted
                  }}>
                    Thoughts on AI, research, and engineering
                  </p>
                </div>
              </div>
              <div style={{display: 'flex', gap: activeSection === 'photography' ? '1rem' : '1.5rem', alignItems: 'center'}}>
                {activeSection !== 'post-detail' && (
                  <nav style={{display: 'flex', gap: activeSection === 'photography' ? '1.1rem' : '1.5rem'}}>
                    {['posts', 'photography', 'about'].map((section) => (
                      <button
                        key={section}
                        onClick={() => {
                          const newSection = section as Section;
                          setActiveSection(newSection);
                          window.history.pushState({ section: newSection, post: null }, '', newSection === 'posts' ? '/' : `#${section}`);
                        }}
                        style={{
                          fontFamily: 'IBM Plex Mono',
                          fontSize: '0.95rem',
                          color: activeSection === section 
                            ? (isDark ? darkText : lightText)
                            : (isDark ? darkMuted : lightMuted),
                          backgroundColor: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          transition: 'color 0.2s',
                          fontWeight: activeSection === section ? 600 : 400
                        }}
                        onMouseOver={(e) => (e.target as HTMLElement).style.color = isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange}
                        onMouseOut={(e) => (e.target as HTMLElement).style.color = activeSection === section 
                          ? (isDark ? darkText : lightText)
                          : (isDark ? darkMuted : lightMuted)}
                      >
                        {section}
                      </button>
                    ))}
                  </nav>
                )}
                {activeSection === 'post-detail' && (
                  <button
                    onClick={navigateBack}
                    style={{
                      fontFamily: 'IBM Plex Mono',
                      fontSize: '0.95rem',
                      color: isDark ? darkMuted : lightMuted,
                      backgroundColor: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'color 0.2s'
                    }}
                    onMouseOver={(e) => (e.target as HTMLElement).style.color = isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange}
                    onMouseOut={(e) => (e.target as HTMLElement).style.color = isDark ? darkMuted : lightMuted}
                  >
                    ← back to posts
                  </button>
                )}
                <button
                  onClick={() => setIsDark(!isDark)}
                  style={{
                    backgroundColor: 'transparent',
                    borderWidth: '1px',
                    borderColor: isDark ? darkDimmed : lightDimmed,
                    color: isDark ? darkText : lightText,
                    padding: '0.375rem 0.75rem',
                    cursor: 'pointer',
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '0.9rem',
                    border: `1px solid ${isDark ? darkDimmed : lightDimmed}`,
                    transition: 'background-color 0.2s',
                    whiteSpace: 'nowrap',
                    borderRadius: '4px'
                  }}
                  onMouseOver={(e) => (e.target as HTMLElement).style.backgroundColor = isDark ? kanagawa.dragon.bgSofter : kanagawa.lotus.bgSofter}
                  onMouseOut={(e) => (e.target as HTMLElement).style.backgroundColor = 'transparent'}
                >
                  {isDark ? '☀ light' : '🌙 dark'}
                </button>
              </div>
            </div>
          </header>

          {/* Posts Section */}
          {activeSection === 'posts' && (
            <div>
              {/* Recent Posts */}
              {recentPosts.map((post) => (
                <article 
                  key={post.slug} 
                  style={{
                    marginBottom: '4rem',
                    paddingBottom: '4rem',
                    borderBottomWidth: '1px',
                    borderBottomColor: isDark ? darkBorder : lightBorder,
                    borderBottomStyle: 'solid',
                    cursor: 'pointer'
                  }}
                  onClick={() => navigateToPost(post)}
                >
                  <div style={{
                    marginBottom: '0.75rem',
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '0.75rem',
                    letterSpacing: '0.05em',
                    color: isDark ? darkMuted : lightMuted
                  }}>
                    {post.date}
                  </div>
                  <h2 style={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '1.3rem',
                    fontWeight: 600,
                    marginBottom: '1rem',
                    lineHeight: 1.4,
                    transition: 'color 0.2s',
                    color: isDark ? kanagawa.dragon.yellow : kanagawa.lotus.yellow
                  }}
                  onMouseOver={(e) => (e.target as HTMLElement).style.color = isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange}
                  onMouseOut={(e) => (e.target as HTMLElement).style.color = isDark ? kanagawa.dragon.yellow : kanagawa.lotus.yellow}
                  >
                    {post.title}
                  </h2>
                  <p style={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '0.95rem',
                    lineHeight: 1.7,
                    marginBottom: '1rem',
                    color: isDark ? darkText : lightText
                  }}>
                    {post.excerpt}
                  </p>
                  <div style={{display: 'flex', gap: '1rem', fontFamily: 'IBM Plex Mono', fontSize: '0.85rem', flexWrap: 'wrap'}}>
                    {post.tags.map((tag) => (
                      <span key={tag} style={{color: isDark ? darkMuted : lightMuted}}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </article>
              ))}

              {/* Archive Section */}
              {archivePosts.length > 0 && (
                <div style={{
                  marginTop: '3rem',
                  paddingTop: '3rem',
                  borderTopWidth: '2px',
                  borderTopColor: isDark ? darkBorder : lightBorder,
                  borderTopStyle: 'solid'
                }}>
                  <h2 style={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '1.5rem',
                    fontWeight: 600,
                    marginBottom: '2rem',
                    color: isDark ? darkText : lightText
                  }}>
                    Archive
                  </h2>
                  <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
                    {archivePosts.map((post) => (
                      <div
                        key={post.slug}
                        style={{
                          cursor: 'pointer',
                          display: 'flex',
                          gap: '1.5rem',
                          alignItems: 'baseline'
                        }}
                        onClick={() => navigateToPost(post)}
                      >
                        <span style={{
                          fontFamily: 'IBM Plex Mono',
                          fontSize: '0.85rem',
                          color: isDark ? darkMuted : lightMuted,
                          minWidth: '100px'
                        }}>
                          {post.date}
                        </span>
                        <span style={{
                          fontFamily: 'IBM Plex Mono',
                          fontSize: '0.95rem',
                          color: isDark ? darkText : lightText,
                          transition: 'color 0.2s'
                        }}
                        onMouseOver={(e) => (e.target as HTMLElement).style.color = isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange}
                        onMouseOut={(e) => (e.target as HTMLElement).style.color = isDark ? darkText : lightText}
                        >
                          {post.title}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Post Detail Section */}
          {activeSection === 'post-detail' && currentPost && (
            <article>
              <div style={{
                marginBottom: '0.75rem',
                fontFamily: 'IBM Plex Mono',
                fontSize: '0.75rem',
                letterSpacing: '0.05em',
                color: isDark ? darkMuted : lightMuted
              }}>
                {currentPost.date}
              </div>
              <h1 style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: '1.75rem',
                fontWeight: 600,
                marginBottom: '2rem',
                lineHeight: 1.3,
                color: isDark ? kanagawa.dragon.yellow : kanagawa.lotus.yellow
              }}>
                {currentPost.title}
              </h1>
              <div 
                className="math-content"
                style={{marginBottom: '2rem'}}
                dangerouslySetInnerHTML={{__html: renderPostContent(currentPost.content)}}
              />
              <div style={{
                display: 'flex', 
                gap: '1rem', 
                fontFamily: 'IBM Plex Mono', 
                fontSize: '0.85rem',
                paddingTop: '2rem',
                borderTop: `1px solid ${isDark ? darkBorder : lightBorder}`,
                flexWrap: 'wrap'
              }}>
                {currentPost.tags.map((tag) => (
                  <span key={tag} style={{color: isDark ? darkMuted : lightMuted}}>
                    {tag}
                  </span>
                ))}
              </div>
            </article>
          )}

          {/* About Section */}
          {activeSection === 'about' && (
            <div>
              <p style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: '0.95rem',
                lineHeight: 1.7,
                color: isDark ? darkText : lightText
              }}>
                I build production GenAI systems. Background in physics from WSU, graduate AI coursework from Stanford. Most of my work is figuring out how to take interesting research and turn it into reliable systems - currently focused on RAG pipelines, LLM evaluation, and deploying models that actually work at scale. I write about the engineering decisions that don't fit in academic papers.
              </p>
            </div>
          )}

          {activeSection === 'photography' && (
            <section className="photography-shell">
              <div className="photography-rail">
                <p className="photography-kicker" style={{margin: 0}}>Photography Archive</p>
                <p className="photography-body" style={{margin: 0, fontSize: '0.72rem', opacity: 0.78}}>
                  {archiveStart} to {archiveEnd} / {photoCount} frames / {archivedLocations} tagged places
                </p>
              </div>

              <div className="photography-stage">
                <div className="photography-frame-panel">
                  <div className="photography-frame">
                    {selectedPhoto.image ? (
                      <img
                        src={selectedPhoto.image}
                        alt={selectedPhoto.title}
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'contain',
                          display: 'block',
                          filter: isDark ? 'contrast(1.02) saturate(0.96)' : 'contrast(1.03) saturate(1.01)'
                        }}
                      />
                    ) : (
                      <div style={{
                        width: '100%',
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontFamily: 'IBM Plex Mono',
                        fontSize: '0.88rem',
                        letterSpacing: '0.14em',
                        textTransform: 'uppercase',
                        color: isDark ? '#F5F1E8' : '#23252B'
                      }}>
                        Add Photo
                      </div>
                    )}
                  </div>
                  <div className="photography-frame-rail">
                    <p className="photography-kicker" style={{margin: 0}}>
                      Frame {String(selectedPhotoIndex + 1).padStart(2, '0')}
                    </p>
                    <p className="photography-body" style={{margin: 0, fontSize: '0.72rem', opacity: 0.72}}>
                      {selectedPhotoLocation ? `${selectedPhotoLocation} / ${selectedPhoto.date}` : selectedPhoto.date}
                    </p>
                  </div>
                </div>

                <aside className="photography-meta-panel">
                  <div className="photography-meta-block">
                    <h3 className="photography-stage-title">{selectedPhoto.title}</h3>
                    <div className="photography-detail-list">
                      {selectedPhotoLocation && (
                        <p style={{
                          fontFamily: 'IBM Plex Mono',
                          fontSize: '0.68rem',
                          fontWeight: 600,
                          letterSpacing: '0.1em',
                          textTransform: 'uppercase',
                          color: isDark ? darkMuted : lightMuted,
                          margin: 0
                        }}>
                          {selectedPhotoLocation}
                        </p>
                      )}
                      <p style={{
                        fontFamily: 'IBM Plex Mono',
                        fontSize: '0.68rem',
                        fontWeight: 600,
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        color: isDark ? darkMuted : lightMuted,
                        margin: 0
                      }}>
                        Captured {selectedPhoto.date}
                      </p>
                    </div>
                    <p className="photography-body" style={{margin: 0, maxWidth: '42rem'}}>
                      {selectedPhoto.description}
                    </p>
                  </div>

                  <div className="photography-actions">
                    <button
                      className="photography-action"
                      onClick={() => setSelectedPhotoIndex((selectedPhotoIndex - 1 + photography.length) % photography.length)}
                    >
                      Prev
                    </button>
                    <button
                      className="photography-action is-primary"
                      onClick={() => setSelectedPhotoIndex((selectedPhotoIndex + 1) % photography.length)}
                    >
                      Next
                    </button>
                  </div>
                </aside>
              </div>

              <div className="photography-strip-container">
                <p className="photography-kicker" style={{marginTop: 0}}>Browse The Archive</p>
                <div className="photography-strip" ref={photographyStripRef}>
                  {photography.map((photo, index) => (
                    <button
                      key={`${photo.title}-${photo.date}`}
                      className={`photography-thumb${selectedPhotoIndex === index ? ' is-active' : ''}`}
                      onClick={() => setSelectedPhotoIndex(index)}
                    >
                      <div className="photography-thumb-media">
                        {photo.image ? (
                          <img
                            src={photo.image}
                            alt={photo.title}
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover',
                              display: 'block',
                              filter: selectedPhotoIndex === index ? 'none' : 'grayscale(0.4) contrast(0.9)'
                            }}
                          />
                        ) : (
                          <div style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontFamily: 'IBM Plex Mono',
                            fontSize: '0.78rem',
                            letterSpacing: '0.14em',
                            textTransform: 'uppercase',
                            color: isDark ? '#F5F1E8' : '#23252B'
                          }}>
                            {String(index + 1).padStart(2, '0')}
                          </div>
                        )}
                      </div>

                      <div className="photography-thumb-copy">
                        <p className="photography-kicker" style={{margin: 0, fontSize: '0.6rem'}}>
                          Frame {String(index + 1).padStart(2, '0')}
                        </p>
                        <p style={{
                          fontFamily: photographyDisplayFont,
                          fontSize: '1.1rem',
                          lineHeight: 0.98,
                          letterSpacing: '-0.02em',
                          margin: '0.25rem 0 0.3rem 0',
                          color: photographyAccent
                        }}>
                          {photo.title}
                        </p>
                        <p style={{
                          fontFamily: 'IBM Plex Mono',
                          fontSize: '0.65rem',
                          letterSpacing: '0.05em',
                          textTransform: 'uppercase',
                          color: isDark ? darkMuted : lightMuted,
                          margin: 0
                        }}>
                          {getDisplayLocation(photo.location) ?? photo.date}
                        </p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          )}

          {/* Footer */}
          <footer style={{
            marginTop: '5rem',
            borderTopWidth: '1px',
            borderTopColor: isDark ? darkBorder : lightBorder,
            borderTopStyle: 'solid',
            paddingTop: '2.5rem'
          }}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem'}}>
              <div>
                <p style={{
                  fontFamily: 'IBM Plex Mono',
                  fontSize: '0.9rem',
                  color: isDark ? darkMuted : lightMuted,
                  marginBottom: '0.25rem'
                }}>
                  © 2026 Johnny Devriese
                </p>
                <p style={{
                  fontFamily: 'IBM Plex Mono',
                  fontSize: '0.8rem',
                  color: isDark ? darkMuted : lightMuted
                }}>
                  human made
                </p>
              </div>
              <SocialIcons 
                colors={{
                  default: isDark ? darkMuted : lightMuted,
                  hover: isDark ? kanagawa.dragon.orange : kanagawa.lotus.orange
                }}
              />
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}
