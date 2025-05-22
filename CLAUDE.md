# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the **Project Aria Research Proposal** for developing a **Recursive Cognitive Integration Framework (RCIF)**. The project proposes a novel cognitive architecture that integrates Project Aria's egocentric perception capabilities with adaptive self-modifying cognitive processes. The core innovation is a **Cognitive Sparse Encoded Architecture (CSEA)** that implements dynamic cognitive boundary management, recursive self-examination processes, and multi-modal knowledge synthesis.

## Repository Structure

This is a **research proposal phase project** - no actual implementation exists yet. The repository contains:

- **MainDoc/**: Main research proposal documents and drafts
  - `# Project Aria Research Proposal- Recursive Cognitive Integration Framework draft 2.md` (354-line comprehensive proposal)
  - `claude-draft2-additions.md` (336-line implementation guidelines with Java interface examples)
- **aria-init/**: Architectural specifications, diagrams, and cognitive models
  - Mermaid diagrams visualizing cognitive model structure and state transitions
  - HTML files with interactive cognitive integration matrices and component visualizations
- **claudeNotes/**: Supplementary documentation and development notes

## Technical Architecture

### Core Innovation: Meta-Cognitive Feedback Loop
The central innovation enabling continuous self-examination and parameter adaptation:
```
ProcessingOutput → PerformanceEvaluation → ParameterOptimization → ProcessingAdjustment
        ↑                                                                ↓
        └────────────────── Recursive Self-Modification ──────────────────┘
```

### Five-Layer Recursive Architecture
1. **Sensory Input**: Multi-modal data from Project Aria sensors (eye-tracking, RGB/SLAM, IMU, PPG, audio)
2. **Sparse Encoding**: Selective feature activation (5-15% density) mimicking human attention mechanisms
3. **Pattern Recognition**: Temporal-spatial regularities across multiple timescales using Mamba sequence models
4. **Knowledge Synthesis**: Cross-modal information integration with dynamic cognitive boundary management
5. **Meta-Cognitive Feedback**: Continuous self-examination and adaptive parameter adjustment

### Project Aria Sensor Integration
- **Eye-tracking**: Attention modeling and flow state detection
- **RGB/SLAM cameras**: Visual and spatial information processing
- **PPG sensors**: Physiological state monitoring for cognitive state detection
- **IMU**: Motion patterns and contextual understanding
- **Audio**: Environmental and communication context

## Planned Implementation Stack

The implementation will utilize these technologies:

- **Core Architecture**: Python/Java for primary framework implementation
- **Performance-Critical Components**: Mojo and Rust for computationally intensive operations
- **Model Implementation**: PyTorch with metal performance shaders
- **Visualization**: Swift for data visualization and user interfaces
- **Configuration**: YAML for system configuration and cognitive model representation

## Development Commands

### Python Components (Planned)

```bash
# Install dependencies
pip install -r requirements.txt
python -m pip install -e .

# Run tests
python run_tests.py
python tests.py

# Type checking and linting
mypy src/python/
flake8 src/python/
```

### Java Components (Planned)

```bash
# Gradle build
./gradlew build
./gradlew test

# Maven build
mvn -B clean package
mvn -B test
```

### Mojo Components (Planned)

```bash
# Install dependencies 
magic install

# Run Mojo code
magic run mojo main.mojo
```

## Development Guidelines

### Documentation Guidelines

- Write clear, concise documentation with proper markdown formatting
- Include diagrams and visualizations using Mermaid syntax
- Document complex cognitive models thoroughly
- Maintain consistent terminology across all documentation

### Cognitive Model Structure

- Follow the recursive cognitive integration framework
- Implement dynamic cognitive boundary management
- Enable self-modifying parameter systems
- Structure using layers: Understanding → Analysis → Exploration → Reflection → Meta-Observation

### Evaluation Framework

The system will be evaluated through multiple approaches:

- **Predictive Accuracy**: Comparison against ground truth data
- **Adaptive Performance**: Measurement of adaptation rate to novel environments
- **Computational Efficiency**: Sparse activation density analysis
- **Cognitive Flexibility**: Boundary adaptation in response to novel categories
- **Flow State Correlation**: Detection of optimal cognitive states

## Development Process

The project follows a **24-week, three-phase development approach**:

### 1. Foundation Phase (Weeks 1-8)
- Implement CSEA core architecture with sparse encoding mechanisms
- Develop basic pattern recognition algorithms using temporal-spatial models
- Create Project Aria sensor integration (VRS format, MPS components)
- Establish baseline performance metrics and evaluation framework

### 2. Integration Phase (Weeks 9-16)
- Build recursive self-examination processes and meta-cognitive feedback loops
- Implement dynamic cognitive boundary management with adaptive thresholding
- Develop advanced temporal-spatial pattern recognition algorithms
- Create meta-cognitive parameter adjustment mechanisms and visualization tools

### 3. Refinement Phase (Weeks 17-24)
- Optimize performance across computational platforms (quantized model execution)
- Implement advanced cognitive state detection using eye-tracking and PPG data
- Develop comprehensive visualization tools for system state representation
- Complete evaluation framework with predictive accuracy, adaptive performance, and flow state correlation metrics

## Key Implementation Components

### Planned Directory Structure
```
aria-research/
├── core/                      # Core CSEA architecture
│   ├── sensory/               # Multi-modal sensory processing
│   ├── encoding/              # Sparse encoding implementation
│   ├── recognition/           # Pattern recognition system
│   └── metacognitive/         # Meta-cognitive feedback loops
├── models/                    # ML model implementations
│   ├── mamba/                 # Sequence models for temporal patterns
│   ├── attention/             # Eye-tracking based attention models
│   └── physiological/         # PPG-based state detection
├── integration/               # Project Aria integration
│   ├── vrs/                   # VRS data handling
│   ├── mps/                   # Machine Perception Services integration
│   └── sdk/                   # Client SDK implementations
├── visualization/             # Cognitive state visualization tools
├── evaluation/                # Evaluation frameworks
└── docs/                      # Technical documentation
```