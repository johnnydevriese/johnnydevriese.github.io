import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Moon, Sun } from 'lucide-react';

declare global {
  interface Window {
    MathJax: any;
  }
}

export default function PrincipalAI() {
  const [isDark, setIsDark] = useState(false);
  const [activeSection, setActiveSection] = useState('posts');

  useEffect(() => {
    // Load MathJax
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js';
    script.async = true;
    script.onload = () => {
      if (window.MathJax) {
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
  }, [activeSection, isDark]);

  const flexoki = {
    base: {
      black: '#100F0F',
      950: '#1C1B1A',
      900: '#282726',
      850: '#343331',
      800: '#403E3C',
      700: '#575653',
      600: '#6F6E69',
      500: '#878580',
      300: '#B7B5AC',
      200: '#CECDC3',
      150: '#DAD8CE',
      100: '#E6E4D9',
      50: '#F2F0E5',
      paper: '#FFFCF0',
    },
    red: { DEFAULT: '#AF3029', light: '#D14D41' },
    orange: { DEFAULT: '#BC5215', light: '#DA702C' },
    yellow: { DEFAULT: '#AD8301', light: '#D0A215' },
    green: { DEFAULT: '#66800B', light: '#879A39' },
    cyan: { DEFAULT: '#24837B', light: '#3AA99F' },
    blue: { DEFAULT: '#205EA6', light: '#4385BE' },
    purple: { DEFAULT: '#5E409D', light: '#8B7EC8' },
    magenta: { DEFAULT: '#A02F6F', light: '#CE5D97' },
  };

  const lightBg = flexoki.base.paper;
  const lightText = flexoki.base.black;
  const lightBorder = flexoki.base[100];
  const lightMuted = flexoki.base[600];
  const lightDimmed = flexoki.base[300];

  const darkBg = flexoki.base.black;
  const darkText = flexoki.base.paper;
  const darkBorder = flexoki.base[850];
  const darkMuted = flexoki.base[500];
  const darkDimmed = flexoki.base[700];

  const articles = [
    {
      date: '2025 · 11',
      title: 'Scaling Reasoning: Capacity, Information, and Emergent Behavior',
      content: [
        'As we push models toward complex reasoning tasks, fundamental questions emerge about model capacity. How much capacity does reasoning actually require? What changes in learning dynamics when we move from predicting tokens to solving novel problems? We investigate the relationship between model architecture, training data, and the emergence of reasoning capabilities across a range of model scales.',
        'Recent work suggests that reasoning requires surprisingly little capacity relative to the size of modern models, yet the efficiency of learning remains an open question. Consider the scaling relationship: $$L(N) = aN^{-\\alpha}$$ where $$L$$ is loss, $$N$$ is model size, and $$\\alpha$$ is the scaling exponent. We examine empirical patterns in scaling laws and what they reveal about the information content of reasoning tasks.'
      ],
      tags: ['research', 'scaling', 'reasoning']
    },
    {
      date: '2025 · 10',
      title: 'Infrastructure for Multi-Expert Models: Serving, Training, and Alignment',
      content: [
        'Mixture-of-experts architectures present novel challenges for production systems. Unlike dense models, MoE systems require careful orchestration across multiple dimensions: load balancing at serving time, efficient parameter sharing during training, and ensuring alignment across expert trajectories.',
        'We explore architectural decisions that propagate through the full stack—from communication patterns in distributed training to serving latency patterns in production. Understanding these trade-offs is essential for building systems that are both fast and reliable.'
      ],
      tags: ['engineering', 'systems', 'infrastructure']
    },
    {
      date: '2025 · 09',
      title: 'Parameter-Efficient Adaptation: When Full Fine-Tuning Becomes Wasteful',
      content: [
        'Modern foundation models contain orders of magnitude more parameters than typical fine-tuning datasets provide information. This inefficiency has motivated research into parameter-efficient methods, but fundamental questions remain: how low can we go? Under what conditions do these methods match full fine-tuning?',
        'We examine the relationship between trainable parameter count, dataset size, and learning efficiency across supervised and reinforcement learning settings. Our empirical findings suggest a clearer picture of when these methods enable genuine speedups versus when they impose unnecessary constraints.'
      ],
      tags: ['research', 'efficiency', 'fine-tuning']
    },
    {
      date: '2025 · 08',
      title: 'Evaluation Beyond Benchmarks: Measuring Generalization and Behavioral Safety',
      content: [
        'Standard benchmarks have become insufficient. As models push capabilities forward, we need evaluation frameworks that catch emergent behaviors, measure robustness to distribution shift, and remain predictive of real-world performance. Static benchmarks cannot keep pace.',
        'This article examines different approaches to behavioral evaluation: from adversarial stress-testing to automated red-teaming, and what each reveals about model robustness. We discuss how to design evaluations that remain meaningful as capabilities increase.'
      ],
      tags: ['research', 'evaluation', 'safety']
    }
  ];

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
      `}</style>
      <div style={{
        backgroundColor: isDark ? darkBg : lightBg,
        color: isDark ? darkText : lightText,
        minHeight: '100vh',
        transition: 'background-color 0.3s, color 0.3s'
      }}>
        <div className="grain" style={{filter: isDark ? 'invert(1) brightness(1.1)' : 'invert(0)'}}></div>
        <div className="content-wrapper mx-auto max-w-2xl px-8 py-16 sm:px-6">
          
          {/* Header */}
          <header style={{
            marginBottom: '3rem',
            borderBottomColor: isDark ? darkBorder : lightBorder,
            borderBottomWidth: '1px',
            paddingBottom: '2.5rem'
          }}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem'}}>
              <div style={{display: 'flex', gap: '1rem', alignItems: 'flex-start'}}>
              <div style={{
                width: '64px',
                height: '64px',
                minWidth: '64px',
                borderRadius: '50%',
                backgroundColor: isDark ? flexoki.base[850] : flexoki.base[150],
                border: `2px solid ${isDark ? darkBorder : lightBorder}`,
                backgroundImage: 'url(data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Ccircle cx="50" cy="35" r="20" fill="%23999"/%3E%3Cpath d="M 15 85 Q 15 60 50 60 Q 85 60 85 85" fill="%23999"/%3E%3C/svg%3E)',
                backgroundSize: 'cover',
                backgroundPosition: 'center'
              }}></div>
              <div>
                <h1 style={{fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.375rem', fontFamily: 'IBM Plex Mono'}}>
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
                <nav style={{display: 'flex', gap: '1.5rem'}}>
                  {['posts', 'about'].map((section) => (
                    <button
                      key={section}
                      onClick={() => setActiveSection(section)}
                      style={{
                        fontFamily: 'IBM Plex Mono',
                        fontSize: '0.95rem',
                        color: activeSection === section 
                          ? (isDark ? darkText : lightText)
                          : (isDark ? darkMuted : lightMuted),
                        backgroundColor: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        transition: 'color 0.2s'
                      }}
                      onMouseOver={(e) => e.target.style.color = isDark ? darkText : lightText}
                      onMouseOut={(e) => e.target.style.color = activeSection === section 
                        ? (isDark ? darkText : lightText)
                        : (isDark ? darkMuted : lightMuted)}
                    >
                      {section}
                    </button>
                  ))}
                </nav>
                <button
                  onClick={() => setIsDark(!isDark)}
                  style={{
                    backgroundColor: 'transparent',
                    borderWidth: '1px',
                    borderColor: isDark ? darkDimmed : lightMuted,
                    color: isDark ? darkText : lightText,
                    padding: '0.375rem 0.75rem',
                    cursor: 'pointer',
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '0.9rem',
                    border: `1px solid ${isDark ? darkDimmed : lightMuted}`,
                    transition: 'background-color 0.2s',
                    whiteSpace: 'nowrap'
                  }}
                  onMouseOver={(e) => e.target.style.backgroundColor = isDark ? flexoki.base[850] : flexoki.base[100]}
                  onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                >
                  {isDark ? '☀ light' : '🌙 dark'}
                </button>
              </div>
            </div>
          </header>

          {/* Posts Section */}
          {activeSection === 'posts' && (
            <div style={{display: 'space-y-20'}}>
              {articles.map((article, idx) => (
                <article key={idx} style={{
                  marginBottom: idx === articles.length - 1 ? 0 : '6rem',
                  paddingBottom: idx === articles.length - 1 ? 0 : '6rem',
                  borderBottomWidth: idx === articles.length - 1 ? 0 : '1px',
                  borderBottomColor: isDark ? darkBorder : lightBorder
                }}>
                  <div style={{
                    marginBottom: '0.75rem',
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '0.75rem',
                    letterSpacing: '0.05em',
                    color: isDark ? darkMuted : lightMuted
                  }}>
                    {article.date}
                  </div>
                  <h2 style={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: '1.3rem',
                    fontWeight: 600,
                    marginBottom: '1.5rem',
                    lineHeight: 1.4
                  }}>
                    {article.title}
                  </h2>
                  <div style={{marginBottom: '1.25rem'}} className="math-content">
                    {article.content.map((para, i) => (
                      <p key={i} style={{
                        fontFamily: 'IBM Plex Mono',
                        fontSize: '0.95rem',
                        lineHeight: 1.7,
                        marginBottom: '1rem',
                        color: isDark ? darkText : lightText
                      }}>
                        {para}
                      </p>
                    ))}
                  </div>
                  <div style={{display: 'flex', gap: '1rem', fontFamily: 'IBM Plex Mono', fontSize: '0.85rem'}}>
                    {article.tags.map((tag) => (
                      <span key={tag} style={{color: isDark ? darkMuted : lightMuted}}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
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
                I'm passionate about research and engineering at the frontier of artificial intelligence.
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
            paddingTop: '2.5rem'
          }}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
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
              <div style={{display: 'flex', gap: '1.5rem'}}>
                {[
                  { label: 'Twitter', url: 'https://twitter.com' },
                  { label: 'GitHub', url: 'https://github.com' },
                  { label: 'Email', url: 'mailto:hello@example.com' }
                ].map((social) => (
                  <a
                    key={social.label}
                    href={social.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      fontFamily: 'IBM Plex Mono',
                      fontSize: '0.9rem',
                      color: isDark ? darkMuted : lightMuted,
                      textDecoration: 'none',
                      transition: 'color 0.2s',
                      cursor: 'pointer'
                    }}
                    onMouseOver={(e) => e.target.style.color = isDark ? flexoki.blue.light : flexoki.blue.DEFAULT}
                    onMouseOut={(e) => e.target.style.color = isDark ? darkMuted : lightMuted}
                  >
                    {social.label}
                  </a>
                ))}
              </div>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}