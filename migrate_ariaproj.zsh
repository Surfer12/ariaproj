#!/bin/zsh

echo "🔄 Starting migration of ariaproj directory..."

# Define the base directory
ARIAPROJ_DIR="/Users/ryandavidoates/systems/ariaproj"

# Change to the ariaproj directory
cd "$ARIAPROJ_DIR" || exit 1

# Create new directory structure
echo "📁 Creating new directory structure..."
mkdir -p MainDoc/archive
mkdir -p aria-init/diagrams
mkdir -p aria-init/models
mkdir -p aria-init/archive
mkdir -p claudeNotes/archive
mkdir -p docs
mkdir -p src
mkdir -p archive

# MainDoc migrations
echo "📄 Migrating MainDoc files..."
# Move old drafts to archive
if [[ -f "MainDoc/changed.md" ]]; then
    mv "MainDoc/changed.md" "MainDoc/archive/"
fi
if [[ -f "MainDoc/draft-1-max.md" ]]; then
    mv "MainDoc/draft-1-max.md" "MainDoc/archive/"
fi
if [[ -f "MainDoc/draft-style-match-1.md" ]]; then
    mv "MainDoc/draft-style-match-1.md" "MainDoc/archive/"
fi

# Rename the main proposal if it exists
if [[ -f "MainDoc/# Project Aria Research Proposal- Recursive Cognitive Integration Framework draft 2.md" ]]; then
    mv "MainDoc/# Project Aria Research Proposal- Recursive Cognitive Integration Framework draft 2.md" "MainDoc/Project Aria Research Proposal - Latest Draft.md"
fi

# aria-init migrations
echo "🎨 Migrating aria-init files..."
# Move diagrams
if [[ -f "aria-init/cognitive-gantt.mermaid" ]]; then
    mv "aria-init/cognitive-gantt.mermaid" "aria-init/diagrams/"
fi
if [[ -f "aria-init/cognitive-model-structure.mermaid" ]]; then
    mv "aria-init/cognitive-model-structure.mermaid" "aria-init/diagrams/"
fi
if [[ -f "aria-init/cognitive-state-diagram.mermaid" ]]; then
    mv "aria-init/cognitive-state-diagram.mermaid" "aria-init/diagrams/"
fi
if [[ -f "aria-init/cognitive-domain-integration-inital.html" ]]; then
    mv "aria-init/cognitive-domain-integration-inital.html" "aria-init/diagrams/"
fi
if [[ -f "aria-init/cognitive-matrix.html" ]]; then
    mv "aria-init/cognitive-matrix.html" "aria-init/diagrams/"
fi
if [[ -f "aria-init/dimensional-integration-matrix-inital.html" ]]; then
    mv "aria-init/dimensional-integration-matrix-inital.html" "aria-init/diagrams/"
fi
if [[ -f "aria-init/dimensional-integration-matrix.html" ]]; then
    mv "aria-init/dimensional-integration-matrix.html" "aria-init/diagrams/"
fi
if [[ -f "aria-init/recursive-cognitive-framework-table-inital.html" ]]; then
    mv "aria-init/recursive-cognitive-framework-table-inital.html" "aria-init/diagrams/"
fi
if [[ -f "aria-init/recursive-cognitive-framework-table.html" ]]; then
    mv "aria-init/recursive-cognitive-framework-table.html" "aria-init/diagrams/"
fi

# Move models
if [[ -f "aria-init/aria.md" ]]; then
    mv "aria-init/aria.md" "aria-init/models/"
fi
if [[ -f "aria-init/aria2.md" ]]; then
    mv "aria-init/aria2.md" "aria-init/models/"
fi
if [[ -f "aria-init/aria3.md" ]]; then
    mv "aria-init/aria3.md" "aria-init/models/"
fi
if [[ -f "aria-init/aria5.md" ]]; then
    mv "aria-init/aria5.md" "aria-init/models/"
fi
if [[ -f "aria-init/aria6.md" ]]; then
    mv "aria-init/aria6.md" "aria-init/models/"
fi
if [[ -f "aria-init/aria7.md" ]]; then
    mv "aria-init/aria7.md" "aria-init/models/"
fi
if [[ -f "aria-init/claude1.md" ]]; then
    mv "aria-init/claude1.md" "aria-init/models/"
fi
if [[ -f "aria-init/draftaria4.md" ]]; then
    mv "aria-init/draftaria4.md" "aria-init/models/"
fi
if [[ -f "aria-init/gem1.md" ]]; then
    mv "aria-init/gem1.md" "aria-init/models/"
fi

# claudeNotes migrations
echo "📝 Migrating claudeNotes files..."
# Move older notes to archive
if [[ -f "claudeNotes/claude-2.md" ]]; then
    mv "claudeNotes/claude-2.md" "claudeNotes/archive/"
fi
if [[ -f "claudeNotes/CLAUDE-3.md" ]]; then
    mv "claudeNotes/CLAUDE-3.md" "claudeNotes/archive/"
fi

# Root level file migrations
echo "🗂️ Migrating root level files..."

# Move project documentation to docs
if [[ -f "Cognitive Integration Module for Project.sty" ]]; then
    mv "Cognitive Integration Module for Project.sty" "docs/"
fi
if [[ -f "I'll reflect on this from a functional a.sty" ]]; then
    mv "I'll reflect on this from a functional a.sty" "docs/"
fi
if [[ -f "Project Name: Cognitive Integration Modu.md" ]]; then
    mv "Project Name: Cognitive Integration Modu.md" "docs/"
fi

# Move code/function files to src
if [[ -f "functions.json" ]]; then
    mv "functions.json" "src/"
fi

# Move root level diagrams to aria-init/diagrams
if [[ -f "gaant-aria.mmd" ]]; then
    mv "gaant-aria.mmd" "aria-init/diagrams/"
fi

# Move legacy drafts and Q&A to archive
if [[ -f "draft-2.md" ]]; then
    mv "draft-2.md" "archive/"
fi
if [[ -f "draft-3.md" ]]; then
    mv "draft-3.md" "archive/"
fi
if [[ -f "gpt4-draft1.md" ]]; then
    mv "gpt4-draft1.md" "archive/"
fi
if [[ -f "questions-for-edu.md" ]]; then
    mv "questions-for-edu.md" "archive/"
fi

# Create README files
echo "📋 Creating README files..."

# Root README
cat > README.md << 'EOF'
# Project Aria: Recursive Cognitive Integration Framework

## Overview
This repository contains research, models, and implementation plans for a novel cognitive architecture integrating Project Aria's egocentric perception with adaptive, recursive self-modifying processes.

## Directory Structure

- **MainDoc/**: Main research proposals and implementation guidelines
- **aria-init/**: Diagrams, cognitive models, and visualizations
- **claudeNotes/**: Communication, style, and process notes
- **docs/**: Project summaries and public-facing documentation
- **src/**: Code, functions, and scripts
- **archive/**: Legacy drafts and Q&A

## Getting Started

1. Read `MainDoc/Project Aria Research Proposal.md` for the core research vision
2. Explore `aria-init/diagrams/` for system and cognitive architecture visualizations
3. See `claudeNotes/` for communication and process guidelines

## Latest Updates

- Organized directory structure for better navigation
- Separated diagrams, models, and documentation
- Archived legacy drafts for historical reference

## Contributing

Please maintain the directory structure when adding new files:
- Research documents → `MainDoc/`
- Diagrams/visualizations → `aria-init/diagrams/`
- Cognitive models → `aria-init/models/`
- Process notes → `claudeNotes/`
- Public documentation → `docs/`
- Code/scripts → `src/`
EOF

# aria-init README
cat > aria-init/README.md << 'EOF'
# aria-init

This directory contains the foundational cognitive models, diagrams, and visualizations for the Project Aria framework.

## Structure

- **diagrams/**: Mermaid diagrams, HTML visualizations, and system architecture charts
- **models/**: Markdown files describing cognitive models and theoretical frameworks
- **archive/**: Superseded or experimental models and diagrams

## Key Components

- Cognitive integration models (aria*.md files)
- System architecture diagrams (Mermaid format)
- Interactive visualizations (HTML)
- Framework tables and matrices

See the root README.md for project context.
EOF

# MainDoc README
cat > MainDoc/README.md << 'EOF'
# MainDoc

Contains the core research proposals and implementation guidelines for Project Aria.

## Structure

- **archive/**: Previous drafts and superseded documents
- Main proposal documents (current versions)

## Key Documents

- Project Aria Research Proposal - Latest Draft.md: The primary research document
- claude-draft2-additions.md: Latest additions and refinements

See the root README.md for project context.
EOF

# claudeNotes README
cat > claudeNotes/README.md << 'EOF'
# claudeNotes

Communication guidelines, process notes, and framework documentation for working with Claude.

## Structure

- **archive/**: Previous versions of communication guidelines
- Current process and style documentation

## Key Documents

- CLAUDE.md: Primary communication framework
- Advanced Cognitive Processing Framework.md: Detailed processing guidelines

See the root README.md for project context.
EOF

# docs README
cat > docs/README.md << 'EOF'
# docs

Public-facing documentation, project summaries, and module descriptions.

## Contents

- Project module descriptions
- Style files and templates
- Summary documents

See the root README.md for project context.
EOF

# src README
cat > src/README.md << 'EOF'
# src

Code, functions, and executable scripts for the Project Aria framework.

## Contents

- Function definitions (JSON format)
- Implementation scripts
- Utility code

See the root README.md for project context.
EOF

# archive README
cat > archive/README.md << 'EOF'
# archive

Legacy drafts, Q&A sessions, and superseded documents.

## Contents

- Previous draft versions
- Historical Q&A sessions
- Experimental documents

These files are preserved for historical reference and context.

See the root README.md for project context.
EOF

echo "✅ Migration completed successfully!"
echo ""
echo "📊 Summary of changes:"
echo "- Organized files into logical directory structure"
echo "- Created comprehensive README files for navigation"
echo "- Separated diagrams, models, and documentation"
echo "- Archived legacy content for historical reference"
echo ""
echo "🔍 Next steps:"
echo "1. Review the new structure: ls -la"
echo "2. Check that all files are in appropriate locations"
echo "3. Customize README files as needed"
echo "4. Remove this migration script when satisfied: rm migrate_ariaproj.zsh"