import { useState, useEffect } from 'react';
import { posts, Post } from './data/posts';
import { renderMarkdown } from './utils/markdown';
import { SocialIcons } from './components/SocialIcons';
import './highlight-theme.css';

declare global {
  interface Window {
    MathJax: any;
  }
}

type Section = 'posts' | 'about' | 'post-detail';

export default function Blog() {
  const [isDark, setIsDark] = useState(() => {
    // Check system preference on initial load
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [activeSection, setActiveSection] = useState<Section>('posts');
  const [currentPost, setCurrentPost] = useState<Post | null>(null);

  // Handle browser back/forward buttons
  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      if (event.state) {
        setActiveSection(event.state.section);
        setCurrentPost(event.state.post || null);
      } else {
        setActiveSection('posts');
        setCurrentPost(null);
      }
    };

    window.addEventListener('popstate', handlePopState);
    
    // Set initial state
    if (!window.history.state) {
      window.history.replaceState({ section: 'posts', post: null }, '', '/');
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
    if (window.MathJax) {
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
    return year >= 2020;
  });

  const archivePosts = posts.filter(post => {
    const year = new Date(post.dateObj).getFullYear();
    return year < 2020;
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
        <div className="content-wrapper" style={{maxWidth: '42rem', margin: '0 auto', padding: '4rem 2rem'}}>
          
          {/* Header */}
          <header style={{
            marginBottom: '3rem',
            borderBottomColor: isDark ? darkBorder : lightBorder,
            borderBottomWidth: '1px',
            paddingBottom: '2.5rem'
          }}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem'}}>
              <div style={{display: 'flex', gap: '1rem', alignItems: 'flex-start'}}>
                <div style={{
                  width: '64px',
                  height: '64px',
                  minWidth: '64px',
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
                      fontSize: '1.5rem', 
                      fontWeight: 700, 
                      marginBottom: '0.375rem', 
                      fontFamily: 'IBM Plex Mono',
                      cursor: activeSection === 'post-detail' ? 'pointer' : 'default',
                      color: isDark ? kanagawa.dragon.red : kanagawa.lotus.red
                    }}
                    onClick={() => activeSection === 'post-detail' && navigateBack()}
                  >
                    Johnny Devriese
                  </h1>
                  <p style={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '0.875rem',
                    color: isDark ? darkMuted : lightMuted
                  }}>
                    Thoughts on AI, research, and engineering
                  </p>
                </div>
              </div>
              <div style={{display: 'flex', gap: '1.5rem', alignItems: 'center'}}>
                {activeSection !== 'post-detail' && (
                  <nav style={{display: 'flex', gap: '1.5rem'}}>
                    {['posts', 'about'].map((section) => (
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
                marginBottom: '1rem',
                color: isDark ? darkText : lightText
              }}>
                I have a background in physics from WSU and have taken graduate courses in AI from Stanford. I'm passionate about applying frontier AI research to solve real-world business problems.
              </p>
              <p style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: '0.95rem',
                lineHeight: 1.7,
                color: isDark ? darkText : lightText
              }}>
                I explore fundamental questions about model scaling, efficiency, evaluation, and safety across a range of scales and modalities.
              </p>
            </div>
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
                  © 2025 Johnny Devriese
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
