---
categories: macos productivity tools
date: '2024-11-01'
layout: post
slug: my-macos-setup
title: My macOS Setup
---

## My macOS Setup

This is my current macOS development setup that I've refined over the years to maximize productivity and maintain a clean workflow.

### Essential Apps

#### Development Tools
- **VS Code** - My primary code editor with GitHub Copilot integration
- **iTerm2** - Superior terminal emulator with split panes and customization
- **Homebrew** - Package manager for macOS
- **Docker Desktop** - Container management
- **Postman** - API development and testing

#### Productivity
- **Alfred** - Spotlight replacement with powerful workflows
- **Rectangle** - Window management with keyboard shortcuts
- **Bear** - Note-taking with markdown support
- **Things 3** - Task management

#### Utilities
- **1Password** - Password management
- **CleanShot X** - Screenshot and screen recording tool
- **Bartender** - Menu bar organization
- **Monitor Control** - External display brightness control

### Terminal Setup

#### Shell Configuration
I use **zsh** with Oh My Zsh for a better terminal experience:

```bash
# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install powerlevel10k theme
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

#### Useful Aliases
```bash
alias gs="git status"
alias gc="git commit -m"
alias gp="git push"
alias gpl="git pull"
alias ll="ls -la"
alias ..="cd .."
alias ...="cd ../.."
```

### Development Environment

#### Python
```bash
# Install pyenv for Python version management
brew install pyenv

# Add to .zshrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

#### Node.js
```bash
# Install nvm for Node version management
brew install nvm

# Add to .zshrc
export NVM_DIR="$HOME/.nvm"
[ -s "/opt/homebrew/opt/nvm/nvm.sh" ] && \. "/opt/homebrew/opt/nvm/nvm.sh"
```

### macOS System Preferences

#### Keyboard Shortcuts
- **Cmd + Space** - Alfred
- **Cmd + Shift + 5** - CleanShot X
- **Ctrl + Opt + Arrow** - Rectangle window management

#### System Settings
```bash
# Show hidden files in Finder
defaults write com.apple.finder AppleShowAllFiles YES

# Faster key repeat rate
defaults write -g KeyRepeat -int 1
defaults write -g InitialKeyRepeat -int 10

# Disable press-and-hold for accents
defaults write -g ApplePressAndHoldEnabled -bool false

# Show path bar in Finder
defaults write com.apple.finder ShowPathbar -bool true

# Show status bar in Finder
defaults write com.apple.finder ShowStatusBar -bool true
```

### VS Code Extensions

Essential extensions I can't live without:
- GitHub Copilot
- Prettier - Code formatter
- ESLint
- Python
- GitLens
- Docker
- Remote - SSH
- Material Icon Theme

### Homebrew Packages

Core packages I install on every Mac:

```bash
brew install git
brew install node
brew install python
brew install wget
brew install htop
brew install tree
brew install jq
brew install ripgrep
brew install fzf
brew install bat
```

### Backup and Sync

- **iCloud Drive** - Documents and Desktop sync
- **Time Machine** - Full system backups to external drive
- **GitHub** - All code is version controlled
- **Mackup** - Sync application settings

```bash
# Install Mackup
brew install mackup

# Backup settings
mackup backup
```

### Workflow Automation

I use **Keyboard Maestro** for custom automations:
- Text expansion snippets
- Application-specific shortcuts
- Clipboard history management
- Window arrangement presets

### Final Thoughts

This setup has evolved over years of experimentation. The key is finding tools that reduce friction and let you focus on the work that matters. Your setup should work for you, not the other way around.

What's in your macOS setup? Let me know on [Twitter](https://twitter.com)!

---

*Last updated: November 2024*
