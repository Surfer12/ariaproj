# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Fractal Communication Framework (FCF) is a system for recursive thought processing with meta-cognitive monitoring capabilities. It implements a modular architecture for conversational AI applications that can adapt responses through reflection, anchoring, and synthesis while maintaining awareness of cognitive processes.

## Repository Structure

The project has several key components:

- **MainDoc/**: Contains main research proposals and documentation for the Recursive Cognitive Integration Framework
- **aria-init/**: Houses diagrams, specifications, cognitive models, and visualization files
- **claudeNotes/**: Includes supplementary documentation and development notes

## Technical Architecture

The Recursive Cognitive Integration Framework consists of five interconnected layers:

1. **Sensory Input**: Processes multi-modal data from Project Aria sensors
2. **Sparse Encoding**: Implements selective feature activation mimicking human attention
3. **Pattern Recognition**: Identifies temporal-spatial regularities across multiple timescales
4. **Knowledge Synthesis**: Integrates information across modalities and timeframes
5. **Meta-Cognitive Feedback**: Enables continuous self-examination and adaptation

The system follows a recursive computational framework that continuously refines its own operation through self-examination.

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

The project follows a three-phase development approach:

1. **Foundation Phase**:
   - Implement core architecture
   - Develop sparse encoding mechanisms
   - Create basic pattern recognition algorithms
   - Establish baseline performance metrics

2. **Integration Phase**:
   - Build recursive self-examination processes
   - Implement dynamic cognitive boundary management
   - Develop temporal-spatial pattern recognition algorithms
   - Create meta-cognitive parameter adjustment mechanisms

3. **Refinement Phase**:
   - Optimize performance across computational platforms
   - Implement advanced cognitive state detection
   - Develop visualization tools for system state representation
   - Create comprehensive evaluation framework

   Write(file_path: CLAUDE.md)…

  ⎿  Updated CLAUDE.md with 80 additions and 89 removals
       1  # CLAUDE.md
       2 
       3  This file provides guidance to Claude Code when working with code in this repository.
       3  This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
       4 
       5  ## Project Overview
       6 
       7  This repository contains the Project Aria Research Proposal for a Recursive Cognitive Integration Framework (RCIF). The framework aims to develop a novel 
     cognitive architecture that integrates Project Aria's egocentric perception capabilities with adaptive self-modifying cognitive processes. The system implements
     dynamic cognitive boundary management, recursive self-examination processes, and multi-modal knowledge synthesis.
         gnitive architecture that integrates Project Aria's egocentric perception capabilities with adaptive self-modifying cognitive processes. The system implement
         s dynamic cognitive boundary management, recursive self-examination processes, and multi-modal knowledge synthesis.
       7  The Fractal Communication Framework (FCF) is a system for recursive thought processing with meta-cognitive monitoring capabilities. It implements a modular 
     architecture for conversational AI applications that can adapt responses through reflection, anchoring, and synthesis while maintaining awareness of cognitive
     processes.
         architecture for conversational AI applications that can adapt responses through reflection, anchoring, and synthesis while maintaining awareness of cognitiv
         e processes.
       8 
       9  ## Repository Structure
      10 
      11  The repository is organized into several key directories:
      11  The project has several key components:
      12 
      13  - **MainDoc/**: Contains the main research proposal documents and drafts
      14    - Primary proposal files (`# Project Aria Research Proposal.md` and version with draft 2)
      15    - Additional specifications in `claude-draft2-additions.md`
      13  - **MainDoc/**: Contains main research proposals and documentation for the Recursive Cognitive Integration Framework
      14  - **aria-init/**: Houses diagrams, specifications, cognitive models, and visualization files
      15  - **claudeNotes/**: Includes supplementary documentation and development notes
      16 
      17  - **aria-init/**: Contains initial drafts, diagrams, and architectural specifications
      18    - Mermaid diagrams for visualizing cognitive model structure
      19    - HTML files with detailed component tables and integration matrices
      20    - Initial concept documents in Markdown format
      17  ## Technical Architecture
      18 
      19  - **claudeNotes/**: Contains additional notes and documentation related to the project
      20    - Style guides and communication recommendations
      21    - Framework concepts and next steps
      19  The Recursive Cognitive Integration Framework consists of five interconnected layers:
      20 
      21  ## Planned Code Architecture
      21  1. **Sensory Input**: Processes multi-modal data from Project Aria sensors
      22  2. **Sparse Encoding**: Implements selective feature activation mimicking human attention
      23  3. **Pattern Recognition**: Identifies temporal-spatial regularities across multiple timescales
      24  4. **Knowledge Synthesis**: Integrates information across modalities and timeframes
      25  5. **Meta-Cognitive Feedback**: Enables continuous self-examination and adaptation
      26 
      27  Based on the project documentation, the implementation architecture will have the following components:
      27  The system follows a recursive computational framework that continuously refines its own operation through self-examination.
      28 
      29  ```
      30  aria-research/
      31  ├── core/                      # Core architecture components
      32  │   ├── sensory/               # Sensory processing modules
      33  │   ├── encoding/              # Sparse encoding implementation
      34  │   ├── recognition/           # Pattern recognition system
      35  │   └── metacognitive/         # Meta-cognitive feedback loops
      36  ├── models/                    # ML model implementations
      37  │   ├── mamba/                 # Sequence models for temporal patterns
      38  │   ├── attention/             # Eye-tracking based attention models
      39  │   └── physiological/         # PPG-based state detection
      40  ├── integration/               # Aria integration components
      41  │   ├── vrs/                   # VRS data handling
      42  │   ├── mps/                   # Machine Perception Services integration
      43  │   └── sdk/                   # Client SDK implementations
      44  ├── visualization/             # Visualization tools
      45  ├── evaluation/                # Evaluation frameworks
      46  └── docs/                      # Documentation
      47  ```
      29  ## Planned Implementation Stack
      30 
      31  ## Planned Technical Stack
      31  The implementation will utilize these technologies:
      32 
      33  - **Core Architecture**: Python/Java for primary framework implementation
      34  - **Performance-Critical Components**: Mojo and Rust for computationally intensive operations
      35  - **Model Implementation**: PyTorch with metal performance shaders for quantized model execution
      35  - **Model Implementation**: PyTorch with metal performance shaders
      36  - **Visualization**: Swift for data visualization and user interfaces
      37  - **Configuration**: YAML for system configuration and cognitive model representation
      38 
      39  ## Development Methodology
      39  ## Development Commands
      40 
      41  The project follows a three-phase development approach:
      41  ### Python Components (Planned)
      42 
      43  1. **Foundation Phase (Weeks 1-8)**
      44     - Implement CSEA core architecture
      45     - Develop sparse encoding mechanisms
      46     - Create basic pattern recognition algorithms
      47     - Establish baseline performance metrics
      43  ```bash
      44  # Install dependencies
      45  pip install -r requirements.txt
      46  python -m pip install -e .
      47 
      48  2. **Integration Phase (Weeks 9-16)**
      49     - Build recursive self-examination processes
      50     - Implement dynamic cognitive boundary management
      51     - Develop temporal-spatial pattern recognition algorithms
      52     - Create meta-cognitive parameter adjustment mechanisms
      48  # Run tests
      49  python run_tests.py
      50  python tests.py
      51 
      52  3. **Refinement Phase (Weeks 17-24)**
      53     - Optimize performance
      54     - Implement advanced cognitive state detection
      55     - Develop visualization tools
      56     - Create comprehensive evaluation framework
      52  # Type checking and linting
      53  mypy src/python/
      54  flake8 src/python/
      55  ```
      56 
      57  ## Key Components
      57  ### Java Components (Planned)
      58 
      59  ### Recursive Cognitive Architecture
      59  ```bash
      60  # Gradle build
      61  ./gradlew build
      62  ./gradlew test
      63 
      64  The architecture consists of five interconnected layers that form a continuous feedback loop:
      65  
      64  # Maven build
      65  mvn -B clean package
      66  mvn -B test
      67  ```
      68  SensoryInput → SparseEncoding → PatternRecognition → PredictiveModeling → KnowledgeSynthesis
      69         ↑                                                                        ↓
      70         └────────────────────── Meta-Cognitive Feedback Loop ────────────────────┘
      71  ```
      68 
      69  ### Meta-Cognitive Feedback Loop
      69  ### Mojo Components (Planned)
      70 
      71  The core innovation enabling continuous self-examination and adaptation:
      71  ```bash
      72  # Install dependencies 
      73  magic install
      74 
      75  # Run Mojo code
      76  magic run mojo main.mojo
      77  ```
      78  ProcessingOutput → PerformanceEvaluation → ParameterOptimization → ProcessingAdjustment
      79          ↑                                                                 ↓
      80          └────────────────── Recursive Self-Modification ──────────────────┘
      81  ```
      78 
      79  ## Implementation Guidelines
      79  ## Development Guidelines
      80 
      81  1. **Sparse Encoding Layer**
      82     - Implement using Java for core architecture with Mojo for performance
      83     - Focus on efficient sparse tensor operations
      84     - Develop adaptive thresholding mechanisms that adjust based on context
      81  ### Documentation Guidelines
      82 
      83  2. **Meta-Cognitive Feedback Loop**
      84     - Design with clear interfaces between components
      85     - Implement parameter tracking to measure adaptation
      86     - Build visualization tools for recursive modification
      83  - Write clear, concise documentation with proper markdown formatting
      84  - Include diagrams and visualizations using Mermaid syntax
      85  - Document complex cognitive models thoroughly
      86  - Maintain consistent terminology across all documentation
      87 
      88  3. **Project Aria Integration**
      89     - VRS Format integration for data handling
      90     - MPS component integration (SLAM, Eye Tracking, etc.)
      91     - Client SDK implementation for data collection
      88  ### Cognitive Model Structure
      89 
      90  ## Evaluation Framework
      90  - Follow the recursive cognitive integration framework
      91  - Implement dynamic cognitive boundary management
      92  - Enable self-modifying parameter systems
      93  - Structure using layers: Understanding → Analysis → Exploration → Reflection → Meta-Observation
      94 
      95  The system will be evaluated through multiple complementary approaches:
      95  ### Evaluation Framework
      96 
      97  - **Predictive Accuracy**: Comparison against ground truth in Aria Digital Twin dataset
      97  The system will be evaluated through multiple approaches:
      98  
      99  - **Predictive Accuracy**: Comparison against ground truth data
     100  - **Adaptive Performance**: Measurement of adaptation rate to novel environments
     101  - **Computational Efficiency**: Sparse activation density analysis
     102  - **Cognitive Flexibility**: Boundary adaptation in response to novel categories
     103  - **Flow State Correlation**: Correlation with self-reported flow experiences
     103  - **Flow State Correlation**: Detection of optimal cognitive states
     104 
     105  ## Documentation Standards
     105  ## Development Process
     106 
     107  - Maintain clear component interface documentation
     108  - Create visual representations of cognitive processes
     109  - Document parameter spaces and integration vectors
     110  - Track system evolution through recursive cycles
     107 \ No newline at end of file
     108  The project follows a three-phase development approach:
     109  
     110  1. **Foundation Phase**:
     111     - Implement core architecture
     112     - Develop sparse encoding mechanisms
     113     - Create basic pattern recognition algorithms
     114     - Establish baseline performance metrics
     115  
     116  2. **Integration Phase**:
     117     - Build recursive self-examination processes
     118     - Implement dynamic cognitive boundary management
     119     - Develop temporal-spatial pattern recognition algorithms
     120     - Create meta-cognitive parameter adjustment mechanisms
     121  
     122  3. **Refinement Phase**:
     123     - Optimize performance across computational platforms
     124     - Implement advanced cognitive state detection
     125     - Develop visualization tools for system state representation
     126     - Create comprehensive evaluation framework
     127 \ No newline at end of file

Tip: Use # to memorize shortcut to quickly add to CLAUDE.md


⏺ I've created a CLAUDE.md file for your Project Aria repository to guide future Claude Code instances. The file provides a comprehensive overview of the Fractal
  Communication Framework (FCF) project, focusing on:

  1. Project overview - explaining the recursive thought processing system with meta-cognitive capabilities
  2. Repository structure - outlining the organization of documentation and specifications
  3. Technical architecture - detailing the five interconnected layers of the cognitive framework
  4. Planned technology stack - covering the multi-language approach (Python/Java core with Mojo/Rust for performance)
  5. Anticipated development commands - for Python, Java, and Mojo components
  6. Development guidelines - for documentation, cognitive model structure, and evaluation
  7. Three-phase development process - Foundation, Integration, and Refinement

  The file will help future Claude Code instances understand your project's purpose and architecture quickly, making them more effective when assisting with
  development.

=================================================================== Previous Conversation Compacted ===================================================================

> This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:
  <analysis>
  Let me analyze the conversation chronologically:

  1. **Initial Request**: The user provided a command-message labeled "init" asking to analyze the codebase and create a CLAUDE.md file that will help future Claude 
  Code instances operate in this repository. The request specifically asked to include common commands, high-level architecture, and to follow certain formatting 
  guidelines.

  2. **My Approach**: I first analyzed the codebase structure and purpose using the Read function to access various files in the repository:
     - I read the main research proposal in `/MainDoc/# Project Aria Research Proposal- Recursive Cognitive Integration Framework draft 2.md`
     - I examined a visualization file in `/aria-init/cognitive-domain-integration-inital.html`
     - I looked at an existing CLAUDE.md in `/Users/ryandavidoates/systems/ariaproj/claudeNotes/CLAUDE.md`
     - I reviewed draft additions in `/MainDoc/claude-draft2-additions.md`
     - I explored cognitive diagrams in `/aria-init/cognitive-state-diagram.mermaid` and `/aria-init/cognitive-model-structure.mermaid`

  3. **Content Creation**: After reviewing these files, I created a comprehensive CLAUDE.md file that included:
     - Project overview describing the Fractal Communication Framework (FCF)
     - Repository structure explanation
     - Technical architecture details of the five interconnected layers
     - Planned implementation stack
     - Development commands for Python, Java, and Mojo components
     - Development guidelines
     - Evaluation framework
     - Three-phase development process

  4. **Action Taken**: I wrote the new CLAUDE.md file to `/Users/ryandavidoates/systems/ariaproj/CLAUDE.md` using the Write function.

  5. **Current Request**: The user has now asked for a detailed summary of the conversation, focusing on capturing technical details, code patterns, and architectural 
  decisions.

  Key files and technical concepts identified:
  - The project is a Recursive Cognitive Integration Framework leveraging Project Aria's sensor capabilities
  - The architecture has five core layers: Sensory Input, Sparse Encoding, Pattern Recognition, Knowledge Synthesis, Meta-Cognitive Feedback
  - Multi-language approach: Python/Java for core architecture, Mojo/Rust for performance-critical components
  - Three-phase development approach: Foundation, Integration, Refinement
  - The system includes a meta-cognitive feedback loop for self-examination and adaptation
  - Various visualization files use HTML and Mermaid syntax for architectural diagrams
  </analysis>

  <summary>
  1. Primary Request and Intent:
     The user's primary request was to analyze the Project Aria codebase and create a CLAUDE.md file to guide future Claude Code instances. This included specific 
  instructions to:
     - Include common development commands (build, lint, test)
     - Document high-level code architecture and structure
     - Present a "big picture" view that would require reading multiple files to understand
     - Avoid repetition and obvious instructions
     - Include important content from README.md and any existing rules files
     - Prefix the file with specific text indicating it's for Claude Code

     Later, the user requested a detailed summary of our conversation, with special focus on technical details, code patterns, and architectural decisions essential 
  for continuing development.

  2. Key Technical Concepts:
     - **Recursive Cognitive Integration Framework (RCIF)**: A system that combines Project Aria's perception capabilities with adaptive cognitive architecture
     - **Cognitive Sparse Encoded Architecture (CSEA)**: The core architecture implementing sparse encoding techniques
     - **Meta-Cognitive Feedback Loop**: The central innovation enabling continuous self-examination and adaptation
     - **Dynamic Cognitive Boundary Management**: Flexible perceptual-conceptual boundaries that adapt based on sensory input
     - **Multi-modal Knowledge Synthesis**: Creating understanding across sensory modalities through sparse encoding
     - **Flow State Detection**: Using eye-tracking and physiological data to detect optimal cognitive states
     - **Sparse Encoding Layer**: Selectively activating features (5-15%) based on context and relevance
     - **Aria Sensor Integration**: Utilizing eye-tracking, RGB/SLAM cameras, IMU, PPG, and audio sensors
     - **Three-Phase Development Methodology**: Foundation, Integration, and Refinement phases

  3. Files and Code Sections:
     - **/MainDoc/# Project Aria Research Proposal- Recursive Cognitive Integration Framework draft 2.md**
       - Central research proposal containing the project's theoretical foundation, objectives, and methodology
       - Defines the five interconnected layers of the architecture and meta-cognitive feedback loop
       - Outlines implementation plan with technical stack and development phases
     
     - **/aria-init/cognitive-domain-integration-inital.html**
       - Visualization of the multi-dimensional cognitive integration matrix
       - Shows cross-domain influence patterns between memory, bias, framework, thought, emotion, and behavior domains
       - Contains integrated system diagram showing information flow between cognitive components
     
     - **/claudeNotes/CLAUDE.md**
       - Previous guidance document with basic development guidelines
       - Lists project structure, documentation guidelines, and cognitive model structure
       - Provided foundation for expanded CLAUDE.md file
     
     - **/MainDoc/claude-draft2-additions.md**
       - Implementation recommendations for the project
       - Contains detailed directory structure, technical implementation guidelines, MPS integration
       - Includes development phase breakdown and evaluation framework implementation
     
     - **/aria-init/cognitive-state-diagram.mermaid**
       - State diagram showing flow between cognitive processing stages
       - Details transitions from sensory perception through meta-cognition to self-modification
       - Annotated with notes explaining key components and processes
     
     - **/aria-init/cognitive-model-structure.mermaid**
       - Graph diagram showing relationships between cognitive components
       - Maps connections between memory, bias, framework, thought, emotion, behavior, and meta-observation
       - Includes data structure details for each component
     
     - **/CLAUDE.md** (Created)
       ```markdown
       # CLAUDE.md
       
       This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
       
       ## Project Overview
       
       The Fractal Communication Framework (FCF) is a system for recursive thought processing with meta-cognitive monitoring capabilities. It implements a modular 
  architecture for conversational AI applications that can adapt responses through reflection, anchoring, and synthesis while maintaining awareness of cognitive 
  processes.
       
       ## Repository Structure
       ...
       ```
       - Comprehensive guidance document for future Claude Code instances
       - Integrates information from multiple source files to provide context
       - Includes planned development commands, technical architecture, and implementation stack
       - Created to fulfill the user's primary request

  4. Problem Solving:
     The main challenge was synthesizing information from multiple complex technical documents into a cohesive, non-repetitive guidance file. I addressed this by:
     - Identifying the core architectural concepts across multiple files
     - Extracting planned implementation details from proposal documents
     - Distinguishing between theoretical concepts and practical implementation components
     - Organizing information into a structured format with clear sections
     - Including anticipated development commands based on the planned technical stack

  5. Pending Tasks:
     - No specific pending tasks were identified beyond creating the CLAUDE.md file and providing this summary.

  6. Current Work:
     I just completed creating the CLAUDE.md file for the Project Aria repository based on the analysis of multiple source files. The file provides comprehensive 
  guidance for future Claude Code instances working with the Recursive Cognitive Integration Framework. The document includes:
     - Project overview explaining the Fractal Communication Framework concept
     - Repository structure documentation
     - Technical architecture details of the five interconnected layers
     - Planned implementation technology stack
     - Development commands for Python, Java, and Mojo components
     - Development guidelines for documentation and cognitive model structure
     - Evaluation framework approach
     - Three-phase development process breakdown

     The completed CLAUDE.md file was written to `/Users/ryandavidoates/systems/ariaproj/CLAUDE.md` and confirmation of the write operation was received.

  7. Optional Next Step:
     As the primary task of creating the CLAUDE.md file has been completed, and the user has explicitly requested a summary of our conversation, no further immediate 
  next steps are required unless the user provides additional requests.
  </summary>.