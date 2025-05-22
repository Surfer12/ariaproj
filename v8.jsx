import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Compass, ZoomIn, ZoomOut, RefreshCw, Info, Eye, Brain, Sparkles, Lightbulb, Book, Users, Activity, BarChart, Network, GitMerge, Share2, GitBranch, GitCommit, Zap } from 'lucide-react';

const RecursiveResponseFramework = () => {
  // State management
  const [activeLayer, setActiveLayer] = useState('foundation');
  const [activeQuestion, setActiveQuestion] = useState(0);
  const [zoomLevel, setZoomLevel] = useState('meso');
  const [showInfo, setShowInfo] = useState(false);
  const [visualizationMode, setVisualizationMode] = useState('cognitive');
  const [narrativeGeneration, setNarrativeGeneration] = useState('');
  const [educationalMode, setEducationalMode] = useState(false);
  const [learningLevel, setLearningLevel] = useState(1);
  const [activeTutorial, setActiveTutorial] = useState(null);
  const [insightLog, setInsightLog] = useState([]);
  const [sparseEncodingRatio, setSparseEncodingRatio] = useState(0.15);
  const [focusPoint, setFocusPoint] = useState({ x: 0.5, y: 0.5 });
  const [recognizedPatterns, setRecognizedPatterns] = useState([]);
  const [displayGraphData, setDisplayGraphData] = useState({ nodes: [], edges: [] });
  const [collaborativeMode, setCollaborativeMode] = useState(false);
  const [collaborators, setCollaborators] = useState([]);
  const [sharedInsights, setSharedInsights] = useState([]);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [audioEngine, setAudioEngine] = useState(null);
  const [attentionalParameters, setAttentionalParameters] = useState({
    baseActivationRatio: 0.15,
    attentionalDecayRate: 5.0,
    semanticWeighting: 0.7,
    temporalPersistence: 0.3
  });
  const [systemMetrics, setSystemMetrics] = useState({
    attentionalEfficiency: 0.85,
    boundaryFlexibility: 0.70,
    recursiveDepth: 2,
    integrationCoherence: 0.80,
    patternRecognitionAccuracy: 0.75,
    crossModalSynthesis: 0.65,
    isomorphicPatternDetection: 0.60,
    semanticNetworkDensity: 0.40,
    collaborativeResonance: 0.50,
    lastEvaluationTimestamp: Date.now()
  });
  const [interactionHistory, setInteractionHistory] = useState([]);
  
  // Core cognitive elements that inform the response structure
  const therapeuticElements = {
    foundation: {
      name: "Grounding",
      description: "Establishing perceptual awareness and safety container",
      color: "#4CAF50",
      symbol: "⊕"
    },
    integration: {
      name: "Integration",
      description: "Connecting insights across domains of experience",
      color: "#9C27B0",
      symbol: "∞"
    },
    transformation: {
      name: "Transformation",
      description: "Disrupting established patterns to enable new possibilities",
      color: "#FF9800",
      symbol: "Δ"
    },
    meta_awareness: {
      name: "Meta-Awareness",
      description: "Recursive self-examination of cognitive processes",
      color: "#00BCD4",
      symbol: "◎"
    }
  };
  
  // Enhanced layers of the response framework with fractal properties
  const responseLayers = {
    foundation: {
      name: "Foundational Layer",
      description: "Project Identification & Integration",
      color: therapeuticElements.foundation.color,
      element: "foundation",
      fractal_property: "Self-similarity across scales",
      attractor_basin: { re: -0.8, im: 0.1 },
      temporal_scale: "immediate",
      bifurcation_point: 3.2,
      learning_sequence: [
        "Identify core perceptual elements",
        "Establish stable representational container",
        "Develop grounded awareness of basic patterns",
        "Connect foundational elements to higher structures"
      ],
      isomorphic_patterns: [
        "Perceptual boundaries create conceptual containers",
        "Attentional selection operates across all cognitive domains",
        "Stable identification precedes flexible transformation"
      ]
    },
    integration: {
      name: "Meta-Cognitive Layer",
      description: "Development Process & Adaptation",
      color: therapeuticElements.integration.color,
      element: "integration",
      fractal_property: "Iteration and refinement",
      attractor_basin: { re: -0.7, im: 0.3 },
      temporal_scale: "medium-term",
      bifurcation_point: 3.5,
      learning_sequence: [
        "Recognize relationships between distinct elements",
        "Identify isomorphic patterns across domains",
        "Practice cross-modal synthesis techniques",
        "Build coherent knowledge structures from fragments"
      ],
      isomorphic_patterns: [
        "Boundary navigation creates integrative tension",
        "Cross-domain synthesis follows consistent patterns",
        "Networks form through repetitive connection cycles"
      ]
    },
    transformation: {
      name: "Emergent Integration Layer",
      description: "Technical Challenges & Solutions",
      color: therapeuticElements.transformation.color,
      element: "transformation",
      fractal_property: "Sensitivity to initial conditions",
      attractor_basin: { re: -0.5, im: 0.5 },
      temporal_scale: "ongoing",
      bifurcation_point: 3.7,
      learning_sequence: [
        "Identify critical bifurcation points",
        "Practice small interventions with large effects",
        "Experience qualitative state transitions", 
        "Develop adaptive responses to emerging patterns"
      ],
      isomorphic_patterns: [
        "Critical points exist in all transformative processes",
        "Phase transitions follow universal mathematical patterns",
        "Small intentional changes create non-linear effects"
      ]
    },
    meta_awareness: {
      name: "Transformative Integration Layer",
      description: "Anthropic API Application",
      color: therapeuticElements.meta_awareness.color,
      element: "meta_awareness",
      fractal_property: "Edge of chaos dynamics",
      attractor_basin: { re: -0.1, im: 0.7 },
      temporal_scale: "recursive",
      bifurcation_point: 3.9,
      learning_sequence: [
        "Develop awareness of your own awareness",
        "Observe patterns in your observation process",
        "Practice recursive self-examination",
        "Generate novel insights through meta-level exploration"
      ],
      isomorphic_patterns: [
        "Self-reference creates new levels of organization",
        "Meta-processes exhibit similar dynamics across domains",
        "Awareness of process transforms the process itself"
      ]
    }
  };

  // Processing level definitions
  const processingLevels = {
    micro: {
      description: "Element-level analysis",
      focus: "Individual components and specific details",
      learning_insights: "Focusing on micro-level details helps identify fundamental building blocks and fine-grained patterns that might be missed at higher levels."
    },
    meso: {
      description: "Pattern-level analysis",
      focus: "Relationships between elements and local context",
      learning_insights: "The meso level reveals how elements connect and form meaningful patterns, highlighting the organizational principles that govern relationships."
    },
    macro: {
      description: "System-level analysis",
      focus: "Overall structure and broader implications",
      learning_insights: "Macro analysis reveals emergent properties that aren't visible at lower levels, showing how local patterns contribute to global structure."
    },
    meta: {
      description: "Process-level analysis",
      focus: "Examination of the analysis process itself",
      learning_insights: "Meta-cognitive awareness allows you to reflect on how you're processing information, enabling adjustments to your own learning approach."
    }
  };
  
  // Advanced pattern recognition and analysis systems
  
  // Isomorphic Pattern Recognition System
  class IsomorphicPatternRecognizer {
    constructor(initialPatterns = {}) {
      this.patternLibrary = initialPatterns;
      this.detectedPatterns = [];
      this.confidenceThreshold = 0.65;
      this.currentScan = null;
    }
    
    // Set pattern library
    setPatternLibrary(patterns) {
      this.patternLibrary = patterns;
    }
    
    // Scan for patterns between two domains
    scanForPatterns(domainA, domainB, metrics) {
      // Cancel any ongoing scan
      if (this.currentScan) {
        clearTimeout(this.currentScan);
      }
      
      // Initialize detection array
      const detectedPatterns = [];
      
      // Get available patterns for these domains
      const patternsA = this.patternLibrary[domainA] || [];
      const patternsB = this.patternLibrary[domainB] || [];
      
      // Calculate base detection probability based on system metrics
      const baseDetectionProbability = metrics.isomorphicPatternDetection * 
                                       metrics.patternRecognitionAccuracy * 
                                       metrics.attentionalEfficiency;
      
      // Simple pattern matching algorithm
      patternsA.forEach(patternA => {
        patternsB.forEach(patternB => {
          // Calculate pattern similarity (placeholder for more complex algorithm)
          const similarity = this.calculatePatternSimilarity(patternA, patternB);
          
          // Adjust with detection probability
          const detectionConfidence = similarity * baseDetectionProbability;
          
          // If confidence exceeds threshold, add to detected patterns
          if (detectionConfidence > this.confidenceThreshold) {
            detectedPatterns.push({
              patternA,
              patternB,
              domains: [domainA, domainB],
              confidence: detectionConfidence,
              timestamp: Date.now(),
              description: this.generatePatternDescription(patternA, patternB, domainA, domainB)
            });
          }
        });
      });
      
      // Update detected patterns asynchronously to simulate processing time
      this.currentScan = setTimeout(() => {
        this.detectedPatterns = [
          ...this.detectedPatterns,
          ...detectedPatterns
        ].slice(-20); // Keep only recent patterns
        
        this.currentScan = null;
      }, 1000 + Math.random() * 2000); // Random delay for realism
      
      return detectedPatterns.length > 0;
    }
    
    // Calculate pattern similarity (placeholder algorithm)
    calculatePatternSimilarity(patternA, patternB) {
      // In a real implementation, this would use more sophisticated NLP techniques
      // For now, we'll use a simple approach based on word overlap
      
      const wordsA = patternA.toLowerCase().split(/\s+/);
      const wordsB = patternB.toLowerCase().split(/\s+/);
      
      // Count shared words
      const sharedWords = wordsA.filter(word => 
        wordsB.includes(word) && word.length > 3 // Only count significant words
      );
      
      // Calculate Jaccard similarity
      const similarity = sharedWords.length / 
                        (wordsA.length + wordsB.length - sharedWords.length);
      
      // Add some randomness to simulate more complex pattern recognition
      return similarity * 0.7 + Math.random() * 0.3;
    }
    
    // Generate natural language description of the pattern
    generatePatternDescription(patternA, patternB, domainA, domainB) {
      const templates = [
        `An isomorphic relationship exists between "${patternA}" in ${domainA} and "${patternB}" in ${domainB}.`,
        `The pattern "${patternA}" in ${domainA} mirrors "${patternB}" in ${domainB}, creating a cross-domain resonance.`,
        `A structural similarity between "${patternA}" (${domainA}) and "${patternB}" (${domainB}) suggests a deeper unifying principle.`
      ];
      
      return templates[Math.floor(Math.random() * templates.length)];
    }
    
    // Get currently detected patterns
    getDetectedPatterns() {
      return this.detectedPatterns;
    }
  }
  
  // Dynamic Knowledge Graph System
  class KnowledgeGraphManager {
    constructor() {
      this.nodes = [];
      this.edges = [];
      this.nodeTypes = ['concept', 'insight', 'pattern', 'question', 'response'];
      this.edgeTypes = ['contains', 'references', 'contradicts', 'extends', 'isomorph'];
      this.layoutParameters = {
        centerX: 400,
        centerY: 200,
        nodeSpacing: 80,
        edgeStrength: 0.5
      };
    }
    
    // Add node to knowledge graph
    addNode(nodeData) {
      // Generate unique ID if not provided
      const id = nodeData.id || `node_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
      
      // Create node with positioning data
      const node = {
        id,
        type: nodeData.type || 'concept',
        label: nodeData.label || 'Untitled Node',
        data: nodeData.data || {},
        x: nodeData.x || this.layoutParameters.centerX + (Math.random() * 200 - 100),
        y: nodeData.y || this.layoutParameters.centerY + (Math.random() * 200 - 100),
        radius: nodeData.radius || 20,
        color: nodeData.color || this.getNodeTypeColor(nodeData.type || 'concept'),
        timestamp: Date.now()
      };
      
      // Add to nodes array
      this.nodes = [...this.nodes, node];
      
      return id;
    }
    
    // Add edge to knowledge graph
    addEdge(sourceId, targetId, type = 'references', strength = 1.0, data = {}) {
      // Generate unique ID
      const id = `edge_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
      
      // Create edge
      const edge = {
        id,
        source: sourceId,
        target: targetId,
        type,
        strength,
        data,
        timestamp: Date.now()
      };
      
      // Add to edges array
      this.edges = [...this.edges, edge];
      
      return id;
    }
    
    // Get color for node type
    getNodeTypeColor(type) {
      const colorMap = {
        concept: '#4285F4',
        insight: '#EA4335',
        pattern: '#FBBC05',
        question: '#34A853',
        response: '#7B1FA2'
      };
      
      return colorMap[type] || '#757575';
    }
    
    // Update node positions based on force-directed layout
    updateLayout() {
      // Simple force-directed layout algorithm
      // In a real implementation, this would use a more sophisticated algorithm
      
      // Apply repulsive forces between nodes
      this.nodes.forEach(node1 => {
        let forceDx = 0;
        let forceDy = 0;
        
        // Repulsive forces between nodes
        this.nodes.forEach(node2 => {
          if (node1.id !== node2.id) {
            const dx = node1.x - node2.x;
            const dy = node1.y - node2.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < this.layoutParameters.nodeSpacing) {
              const repulsiveForce = this.layoutParameters.nodeSpacing / (distance + 0.1);
              forceDx += dx * repulsiveForce * 0.05;
              forceDy += dy * repulsiveForce * 0.05;
            }
          }
        });
        
        // Apply edge forces
        this.edges.forEach(edge => {
          if (edge.source === node1.id || edge.target === node1.id) {
            const otherNodeId = edge.source === node1.id ? edge.target : edge.source;
            const otherNode = this.nodes.find(n => n.id === otherNodeId);
            
            if (otherNode) {
              const dx = otherNode.x - node1.x;
              const dy = otherNode.y - node1.y;
              const distance = Math.sqrt(dx * dx + dy * dy);
              
              const attractiveForce = (distance - this.layoutParameters.nodeSpacing) * edge.strength;
              forceDx += dx * attractiveForce * 0.01;
              forceDy += dy * attractiveForce * 0.01;
            }
          }
        });
        
        // Apply forces to node position
        node1.x += forceDx;
        node1.y += forceDy;
        
        // Keep nodes within bounds
        const bounds = 100;
        node1.x = Math.max(bounds, Math.min(this.layoutParameters.centerX * 2 - bounds, node1.x));
        node1.y = Math.max(bounds, Math.min(this.layoutParameters.centerY * 2 - bounds, node1.y));
      });
    }
    
    // Get knowledge graph
    getGraph() {
      return {
        nodes: this.nodes,
        edges: this.edges
      };
    }
    
    // Create graph from cognitive framework data
    buildFromCognitiveFramework(questions, activeLayer, patternRecognizer) {
      // Start with clean graph
      this.nodes = [];
      this.edges = [];
      
      // Add layer node
      const layerNodeId = this.addNode({
        type: 'concept',
        label: `Layer: ${activeLayer}`,
        color: responseLayers[activeLayer].color,
        data: { layer: activeLayer }
      });
      
      // Add questions and responses
      Object.entries(questions).forEach(([layer, layerQuestions]) => {
        layerQuestions.forEach((question, index) => {
          // Add question node
          const questionNodeId = this.addNode({
            type: 'question',
            label: question.standard,
            data: { 
              layer,
              index,
              standard: question.standard,
              enhanced: question.enhanced
            }
          });
          
          // Add edge from layer to question if this is the active layer
          if (layer === activeLayer) {
            this.addEdge(layerNodeId, questionNodeId, 'contains');
          }
          
          // Add response node
          const responseNodeId = this.addNode({
            type: 'response',
            label: question.response.substring(0, 30) + '...',
            data: {
              layer,
              index,
              response: question.response
            }
          });
          
          // Add edge from question to response
          this.addEdge(questionNodeId, responseNodeId, 'references');
          
          // Add insights from fractal insights
          if (question.fractal_insights) {
            question.fractal_insights.forEach((insight, i) => {
              const insightNodeId = this.addNode({
                type: 'insight',
                label: insight.substring(0, 20) + '...',
                data: {
                  layer,
                  questionIndex: index,
                  insight,
                  insightIndex: i
                }
              });
              
              // Add edge from response to insight
              this.addEdge(responseNodeId, insightNodeId, 'contains');
            });
          }
        });
      });
      
      // Add pattern nodes from pattern recognizer
      if (patternRecognizer) {
        const patterns = patternRecognizer.getDetectedPatterns();
        
        patterns.forEach(pattern => {
          // Add pattern node
          const patternNodeId = this.addNode({
            type: 'pattern',
            label: `Pattern: ${pattern.domains.join('↔')}`,
            data: {
              pattern,
            }
          });
          
          // Find related nodes
          const domainNodes = this.nodes.filter(node => 
            node.data && node.data.layer && pattern.domains.includes(node.data.layer)
          );
          
          // Connect pattern to domain nodes
          domainNodes.forEach(node => {
            this.addEdge(patternNodeId, node.id, 'isomorph', pattern.confidence);
          });
        });
      }
      
      // Run layout algorithm
      for (let i = 0; i < 10; i++) {
        this.updateLayout();
      }
      
      return {
        nodes: this.nodes,
        edges: this.edges
      };
    }
  }
  
  // NEW: KnowledgeGraphRegistry class definition
  class KnowledgeGraphRegistry {
    constructor() {
      this.instances = new Map();
      // console.log("KnowledgeGraphRegistry initialized");
    }

    /**
     * Retrieves an existing instance of KnowledgeGraphManager or creates a new one.
     * @param {string} id - The unique identifier for the KnowledgeGraphManager instance.
     * @param {object} [options={}] - Options to pass to the KnowledgeGraphManager constructor if creating a new instance.
     * @returns {KnowledgeGraphManager} The KGM instance.
     */
    getOrCreateInstance(id, options = {}) {
      if (!this.instances.has(id)) {
        // console.log(`Registry: Creating new KnowledgeGraphManager instance with id "${id}"`);
        // KnowledgeGraphManager class must be in scope here.
        const newInstance = new KnowledgeGraphManager({ ...options, id });
        this.instances.set(id, newInstance);
      }
      return this.instances.get(id);
    }

    /**
     * Retrieves an existing instance of KnowledgeGraphManager.
     * @param {string} id - The unique identifier for the instance.
     * @returns {KnowledgeGraphManager | undefined} The instance or undefined if not found.
     */
    getInstance(id) {
      if (!this.instances.has(id)) {
        // console.warn(`Registry: No KnowledgeGraphManager instance found with id "${id}"`);
      }
      return this.instances.get(id);
    }

    /**
     * Destroys a managed KnowledgeGraphManager instance.
     * @param {string} id - The unique identifier for the instance to destroy.
     * @returns {boolean} True if an instance was destroyed, false otherwise.
     */
    destroyInstance(id) {
      if (this.instances.has(id)) {
        // console.log(`Registry: Destroying KnowledgeGraphManager instance with id "${id}"`);
        this.instances.delete(id);
        return true;
      }
      // console.warn(`Registry: Attempted to destroy non-existent instance with id "${id}"`);
      return false;
    }

    /**
     * Lists the IDs of all currently managed instances.
     * @returns {string[]} An array of instance IDs.
     */
    listInstanceIds() {
      return Array.from(this.instances.keys());
    }
  }
  // END NEW: KnowledgeGraphRegistry class definition
  
  // Collaborative Framework Components
  class CollaborativeFramework {
    constructor() {
      this.collaborators = [];
      this.sharedInsights = [];
      this.activeCollaborations = [];
    }
    
    // Add a collaborator
    addCollaborator(id, name, avatar, role = 'explorer') {
      const collaborator = {
        id,
        name,
        avatar,
        role,
        joinedAt: Date.now(),
        status: 'active',
        focusArea: null,
        lastActivity: Date.now()
      };
      
      this.collaborators = [...this.collaborators, collaborator];
      
      return collaborator;
    }
    
    // Update collaborator status
    updateCollaboratorStatus(id, updates) {
      this.collaborators = this.collaborators.map(collaborator => 
        collaborator.id === id ? {...collaborator, ...updates, lastActivity: Date.now()} : collaborator
      );
    }
    
    // Add shared insight
    addSharedInsight(collaboratorId, content, type = 'observation', referencedElements = []) {
      const collaborator = this.collaborators.find(c => c.id === collaboratorId);
      
      if (!collaborator) return null;
      
      const insight = {
        id: `insight_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
        collaboratorId,
        collaboratorName: collaborator.name,
        content,
        type,
        referencedElements,
        timestamp: Date.now(),
        reactions: [],
        threads: []
      };
      
      this.sharedInsights = [...this.sharedInsights, insight];
      
      return insight;
    }
    
    // Add reaction to insight
    addReactionToInsight(insightId, collaboratorId, reactionType) {
      const insight = this.sharedInsights.find(i => i.id === insightId);
      if (!insight) return false;
      
      const reaction = {
        collaboratorId,
        reactionType,
        timestamp: Date.now()
      };
      
      insight.reactions = [...insight.reactions, reaction];
      
      this.sharedInsights = this.sharedInsights.map(i => 
        i.id === insightId ? insight : i
      );
      
      return true;
    }
    
    // Get insights filtered by criteria
    getFilteredInsights(criteria = {}) {
      let filtered = [...this.sharedInsights];
      
      if (criteria.collaboratorId) {
        filtered = filtered.filter(i => i.collaboratorId === criteria.collaboratorId);
      }
      
      if (criteria.type) {
        filtered = filtered.filter(i => i.type === criteria.type);
      }
      
      if (criteria.referencedElement) {
        filtered = filtered.filter(i => 
          i.referencedElements.includes(criteria.referencedElement)
        );
      }
      
      if (criteria.since) {
        filtered = filtered.filter(i => i.timestamp >= criteria.since);
      }
      
      // Sort by timestamp (newest first)
      filtered.sort((a, b) => b.timestamp - a.timestamp);
      
      return filtered;
    }
    
    // Start a new collaboration on a specific element
    startCollaboration(element, initiatorId) {
      const collaborationId = `collab_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
      
      const collaboration = {
        id: collaborationId,
        element,
        initiatorId,
        startedAt: Date.now(),
        participants: [initiatorId],
        status: 'active',
        focusPoints: [],
        insights: []
      };
      
      this.activeCollaborations = [...this.activeCollaborations, collaboration];
      
      return collaboration;
    }
    
    // Join existing collaboration
    joinCollaboration(collaborationId, collaboratorId) {
      const collaboration = this.activeCollaborations.find(c => c.id === collaborationId);
      if (!collaboration) return false;
      
      if (!collaboration.participants.includes(collaboratorId)) {
        collaboration.participants = [...collaboration.participants, collaboratorId];
        
        this.activeCollaborations = this.activeCollaborations.map(c => 
          c.id === collaborationId ? collaboration : c
        );
      }
      
      return true;
    }
    
    // Add focus point to collaboration
    addFocusPoint(collaborationId, collaboratorId, coordinates, comment = '') {
      const collaboration = this.activeCollaborations.find(c => c.id === collaborationId);
      if (!collaboration) return false;
      
      if (!collaboration.participants.includes(collaboratorId)) {
        this.joinCollaboration(collaborationId, collaboratorId);
      }
      
      const focusPoint = {
        collaboratorId,
        coordinates,
        comment,
        timestamp: Date.now()
      };
      
      collaboration.focusPoints = [...collaboration.focusPoints, focusPoint];
      
      this.activeCollaborations = this.activeCollaborations.map(c => 
        c.id === collaborationId ? collaboration : c
      );
      
      return true;
    }
    
    // Get active collaborations for element
    getCollaborationsForElement(element) {
      return this.activeCollaborations.filter(c => 
        c.element === element && c.status === 'active'
      );
    }
    
    // Get active collaboration data
    getCollaborationData(collaborationId) {
      return this.activeCollaborations.find(c => c.id === collaborationId);
    }
    
    // End collaboration
    endCollaboration(collaborationId) {
      this.activeCollaborations = this.activeCollaborations.map(c => 
        c.id === collaborationId ? {...c, status: 'completed', endedAt: Date.now()} : c
      );
      
      return true;
    }
    
    // Generate summary of collaboration
    generateCollaborationSummary(collaborationId) {
      const collaboration = this.activeCollaborations.find(c => c.id === collaborationId);
      if (!collaboration) return null;
      
      // This would be a more sophisticated summarization algorithm in a real implementation
      return {
        element: collaboration.element,
        participantCount: collaboration.participants.length,
        duration: collaboration.endedAt ? (collaboration.endedAt - collaboration.startedAt) : (Date.now() - collaboration.startedAt),
        insightCount: collaboration.insights.length,
        focusPointCount: collaboration.focusPoints.length,
        summary: `Collaboration on ${collaboration.element} with ${collaboration.participants.length} participants resulted in ${collaboration.insights.length} shared insights.`
      };
    }
  }

  // Educational scaffolding content
  const educationalContent = {
    tutorials: {
      fractal_basics: {
        title: "Understanding Fractal Patterns",
        steps: [
          {
            title: "What are Fractals?",
            content: "Fractals are complex patterns that exhibit self-similarity across different scales. Unlike regular geometric shapes, fractals contain infinite detail that repeats as you zoom in or out.",
            interaction: "Try changing between zoom levels to observe self-similar patterns at different scales."
          },
          {
            title: "The Mandelbrot Formula",
            content: "The formula z = z² + c generates complex fractal patterns. In our cognitive framework, z represents your current understanding, z² represents recursive elaboration, and c introduces new perspectives.",
            interaction: "Switch between visualization modes to see how this formula applies to cognitive processes."
          },
          {
            title: "Bifurcation Points",
            content: "Small changes at critical thresholds (bifurcation points) can create dramatic differences in outcomes. These represent moments where tiny adjustments lead to qualitatively different understandings.",
            interaction: "Examine the bifurcation points in the additional analysis section."
          },
          {
            title: "Edge of Chaos",
            content: "The most generative zone for new insights exists at the boundary between order and chaos - where patterns are neither too rigid nor too random.",
            interaction: "Look for patterns in the fractal visualization that show complex, neither completely ordered nor random structures."
          }
        ]
      },
      cognitive_layers: {
        title: "Exploring Cognitive Layers",
        steps: [
          {
            title: "Grounding (⊕)",
            content: "The foundation layer establishes perceptual awareness and creates a stable container for exploration. It's where we identify core elements and establish basic patterns.",
            interaction: "Select the Foundational Layer to experience how it provides stability for further exploration."
          },
          {
            title: "Integration (∞)",
            content: "The integration layer connects disparate elements into coherent patterns, enabling cross-domain synthesis and revealing relationships between seemingly separate concepts.",
            interaction: "Select the Meta-Cognitive Layer and observe how different questions connect in the Integration Map view."
          },
          {
            title: "Transformation (Δ)",
            content: "The transformation layer disrupts established patterns to enable new possibilities. Small changes at critical points create qualitative shifts in understanding.",
            interaction: "Select the Emergent Integration Layer and observe how different parameter values create dramatically different outcomes."
          },
          {
            title: "Meta-Awareness (◎)",
            content: "The meta-awareness layer enables recursive self-examination of cognitive processes. Here we observe not just content but the process of observation itself.",
            interaction: "Select the Transformative Integration Layer and notice the Meta-Cognitive System Analysis section."
          }
        ]
      },
      recursive_thinking: {
        title: "Developing Recursive Thinking",
        steps: [
          {
            title: "What is Recursion?",
            content: "Recursion occurs when a process refers to itself. In cognitive terms, this happens when we think about our own thinking, creating nested layers of awareness.",
            interaction: "Observe how the system monitors and modifies its own parameters in the Meta-Cognitive System Analysis."
          },
          {
            title: "Depth of Recursion",
            content: "Recursion can occur at multiple depths. First-order recursion is thinking about thinking, second-order is thinking about thinking about thinking, and so on.",
            interaction: "Notice how the zoom levels (especially 'meta') create deeper levels of recursive exploration."
          },
          {
            title: "Self-Reference Without Paradox",
            content: "Unlike logical paradoxes, cognitive recursion can be stable and productive when properly structured. Each recursive layer adds new insights rather than contradictions.",
            interaction: "Look at how fractal insights build upon each other in the Additional Analysis section."
          },
          {
            title: "Practical Applications",
            content: "Recursive thinking enables meta-learning (learning how to learn), improved decision-making through examining thought processes, and enhanced creativity.",
            interaction: "Explore different questions to see how recursive examination creates deeper insights than direct answers."
          }
        ]
      },
      sparse_encoding: {
        title: "Sparse Encoding & Attention",
        steps: [
          {
            title: "Selective Attention",
            content: "The brain doesn't process all information equally. It selectively enhances relevant signals while suppressing irrelevant noise - a form of sparse encoding.",
            interaction: "Adjust the sparse encoding ratio to see how focus changes information density."
          },
          {
            title: "Efficiency Through Constraint",
            content: "Processing only 5-15% of available information actually enhances cognitive efficiency, allowing more resources for the most relevant content.",
            interaction: "Move your cursor over text to shift the focus point and observe how attention highlights different content areas."
          },
          {
            title: "Pattern Recognition",
            content: "Sparse encoding enhances pattern recognition by reducing noise and highlighting structural relationships between key elements.",
            interaction: "Try different sparse encoding ratios to find the optimal balance between focus and context."
          },
          {
            title: "Dynamic Attention Allocation",
            content: "Attention isn't static but constantly shifts based on changing goals, environmental cues, and emerging insights.",
            interaction: "Notice how the system adjusts its attentional efficiency based on your interaction patterns."
          }
        ]
      }
    },
    learning_paths: {
      beginner: {
        description: "Introduction to recursive cognitive concepts",
        recommended_tutorials: ["fractal_basics", "cognitive_layers"],
        starting_layer: "foundation",
        zoom_level: "meso"
      },
      intermediate: {
        description: "Deeper exploration of recursive patterns",
        recommended_tutorials: ["recursive_thinking", "sparse_encoding"],
        starting_layer: "integration",
        zoom_level: "macro"
      },
      advanced: {
        description: "Meta-cognitive practice and application",
        recommended_tutorials: [],
        starting_layer: "meta_awareness",
        zoom_level: "meta"
      }
    }
  };
  
  // Initialize systems
  const [patternRecognizer] = useState(() => new IsomorphicPatternRecognizer());
  // MODIFIED: Use useMemo and the registry to instantiate knowledgeGraphManager
  const registry = useMemo(() => {
    console.log("🔧 KnowledgeGraphRegistry: Creating new registry instance");
    return new KnowledgeGraphRegistry();
  }, []);
  const knowledgeGraphManager = useMemo(() => {
    console.log("📊 KnowledgeGraphRegistry: Getting or creating 'mainAriaGraph' instance");
    const instance = registry.getOrCreateInstance('mainAriaGraph');
    console.log("📊 KnowledgeGraphRegistry: Current instances:", registry.listInstanceIds());
    return instance;
  }, [registry]);
  const [collaborationSystem] = useState(() => new CollaborativeFramework());
  const [bifurcationSystem] = useState(() => new BifurcationAnalysisSystem());
  const [selfModifyingSystem] = useState(() => new SelfModifyingArchitecture());
  const [environmentalSystem] = useState(() => new EnvironmentalContextSystem());
  
  // Initialize on mount
  useEffect(() => {
    // Initialize pattern recognizer with layer patterns
    const patternLibrary = {};
    Object.keys(responseLayers).forEach(layer => {
      patternLibrary[layer] = responseLayers[layer].isomorphic_patterns || [];
    });
    patternRecognizer.setPatternLibrary(patternLibrary);
    
    // Initialize bifurcation analysis
    bifurcationSystem.defineParameterRange('attentionalEfficiency', 0.1, 1.0);
    bifurcationSystem.defineParameterRange('boundaryFlexibility', 0.1, 1.0);
    bifurcationSystem.defineParameterRange('recursiveDepth', 1, 5);
    bifurcationSystem.calculateStateSpace();
    
    // Initialize self-modifying architecture
    selfModifyingSystem.initializeParameterTracking(systemMetrics);
    
    // Define adaptation rules
    selfModifyingSystem.defineAdaptationRule(
      'increase_recursion_on_pattern_detection',
      'recursiveDepth',
      (history, params) => params.patternRecognitionAccuracy > 0.8,
      (history, params, meta) => 0.1 * meta.adaptationRate
    );
    
    selfModifyingSystem.defineAdaptationRule(
      'enhance_flexibility_on_exploration',
      'boundaryFlexibility',
      (history, params) => params.attentionalEfficiency < 0.6,
      (history, params, meta) => 0.05 * meta.adaptationRate
    );
    
    // Initialize environmental system
    environmentalSystem.updateContext();
    
    // Define rhythm patterns
    environmentalSystem.defineRhythmPattern(
      'attentionalOscillation',
      300000, // 5 minute period
      (phase, context) => Math.sin(phase * Math.PI * 2) * 0.05 // Oscillation amplitude
    );
    
    // Define context adaptation profiles
    environmentalSystem.defineAdaptationProfile(
      'morning_focus',
      (context) => context.timeOfDay && context.timeOfDay.phase === 'morning',
      {
        attentionalEfficiency: 0.9,
        boundaryFlexibility: 0.5
      }
    );
    
    environmentalSystem.defineAdaptationProfile(
      'evening_creativity',
      (context) => context.timeOfDay && context.timeOfDay.phase === 'evening',
      {
        boundaryFlexibility: 0.8,
        crossModalSynthesis: 0.8
      }
    );
    
    // Periodically update systems
    const intervalId = setInterval(() => {
      // Update environmental context
      environmentalSystem.updateContext();
      environmentalSystem.updateCognitiveLoad(systemMetrics, interactionHistory);
      
      // Apply contextual adaptations if enabled
      if (systemMetrics.boundaryFlexibility > 0.5) {
        const environmentalParams = environmentalSystem.applyContextualAdaptations(systemMetrics);
        const rhythmicParams = environmentalSystem.applyRhythmicVariations(environmentalParams);
        
        // Update self-modifying system
        selfModifyingSystem.updateParameterHistory(rhythmicParams);
        
        // Apply adaptation rules
        const adaptedParams = selfModifyingSystem.applyAdaptationRules(rhythmicParams);
        
        // Update meta-parameters
        selfModifyingSystem.updateMetaParameters(adaptedParams);
        
        // Generate meta-commentary
        const commentary = selfModifyingSystem.generateMetaCommentary();
        
        // Only update state if significant changes occurred
        if (JSON.stringify(adaptedParams) !== JSON.stringify(systemMetrics)) {
          setSystemMetrics(prev => ({
            ...prev,
            ...adaptedParams
          }));
          
          // Generate narrative about parameter changes
          setNarrativeGeneration(commentary);
          
          // Check for bifurcation points
          const dynamicsAnalysis = bifurcationSystem.analyzeCurrentDynamics();
          if (dynamicsAnalysis.insightPotential > 0.7) {
            // High insight potential detected
            setInsightLog(prev => [
              ...prev,
              {
                type: 'bifurcation_insight',
                message: dynamicsAnalysis.dynamicsDescription,
                timestamp: Date.now()
              }
            ]);
          }
        }
      }
      
      // Run pattern detection periodically
      if (Math.random() < 0.2) { // 20% chance each interval
        const layerKeys = Object.keys(responseLayers);
        if (layerKeys.length >= 2) {
          const randomLayer1 = layerKeys[Math.floor(Math.random() * layerKeys.length)];
          let randomLayer2;
          do {
            randomLayer2 = layerKeys[Math.floor(Math.random() * layerKeys.length)];
          } while (randomLayer1 === randomLayer2);
          
          // Scan for patterns between random layers
          const patternsFound = patternRecognizer.scanForPatterns(
            randomLayer1, 
            randomLayer2, 
            systemMetrics
          );
          
          if (patternsFound) {
            // New patterns detected, update knowledge graph
            const newGraphDataFromManager = knowledgeGraphManager.buildFromCognitiveFramework(
              enhancedQuestions,
              activeLayer,
              patternRecognizer
            );
            
            setDisplayGraphData(newGraphDataFromManager);
            
            // Log insight
            setInsightLog(prev => [
              ...prev,
              {
                type: 'isomorphic_pattern',
                message: `Detected isomorphic patterns between ${randomLayer1} and ${randomLayer2} layers.`,
                timestamp: Date.now()
              }
            ]);
          }
        }
      }
    }, 10000); // Every 10 seconds
    
    return () => clearInterval(intervalId);
  }, []);
  
  // Sparse representation methods - implementing selective feature activation
  const generateDynamicSparseRepresentation = (content, focusPoint, attentionalParameters) => {
    if (!content) return "";
    
    const {
      baseActivationRatio,    // Core activation percentage (0.05-0.15 recommended)
      attentionalDecayRate,   // How quickly attention diminishes with distance
      semanticWeighting,      // How much semantic importance affects activation
      temporalPersistence     // How long previously activated elements remain partially visible
    } = attentionalParameters;
    
    // Parse content into processable elements (words initially, can extend to other structures)
    const words = content.split(' ');
    
    // Generate activation map based on multiple parameters
    const activationValues = words.map((word, i) => {
      // Calculate normalized position (0-1 range)
      const position = i / words.length;
      
      // Calculate spatial attention (distance from focus point)
      const spatialDistance = Math.abs(position - focusPoint.x);
      const spatialAttention = Math.exp(-spatialDistance * attentionalDecayRate);
      
      // Calculate semantic importance (placeholder - would use more sophisticated algorithms)
      const semanticImportance = word.length > 6 ? 1.2 : 
                               (word.endsWith('.') || word.endsWith('?') || word.endsWith('!')) ? 1.3 : 1.0;
      
      // Calculate temporal persistence (placeholder for more advanced implementation)
      const temporalWeight = 1.0; // Would be affected by interaction history
      
      // Combine factors to determine final activation value
      const activationValue = (
        spatialAttention * 0.6 + 
        semanticImportance * semanticWeighting * 0.3 + 
        temporalWeight * temporalPersistence * 0.1
      );
      
      return {
        word,
        activation: activationValue,
        position
      };
    });
    
    // Apply activation threshold with dynamic baseline
    const baselineThreshold = 1 - baseActivationRatio;
    const dynamicThreshold = baselineThreshold * (1 - (systemMetrics.attentionalEfficiency * 0.2));
    
    // Apply representation transformation based on activation values
    const sparseRepresentation = activationValues.map(item => {
      // Determine if element is activated based on threshold and randomness
      const isActivated = 
        item.activation > dynamicThreshold || 
        Math.random() < baseActivationRatio * 0.2; // Small random factor
      
      // Handle partially activated elements (creates gradation rather than binary visibility)
      if (isActivated) {
        return item.word;
      } else if (item.activation > dynamicThreshold * 0.7) {
        // Partially visible (could use styling or fading in real implementation)
        return `<span class="partial-activation">${item.word}</span>`;
      } else {
        // Not activated
        return "···";
      }
    });
    
    return sparseRepresentation.join(' ');
  };
  
  // Simulated meta-cognitive processes for self-examination and parameter adjustment
  const performMetaCognitiveEvaluation = () => {
    // Calculate time since last evaluation
    const currentTime = Date.now();
    const evaluationInterval = currentTime - systemMetrics.lastEvaluationTimestamp;
    
    // Perform evaluation and parameter adjustments based on interaction history
    const adjustments = {
      attentionalEfficiency: Math.min(0.98, systemMetrics.attentionalEfficiency + 
        (interactionHistory.length > 5 ? 0.02 : -0.01)),
      boundaryFlexibility: Math.min(0.95, systemMetrics.boundaryFlexibility + 
        (activeLayer === 'transformation' ? 0.03 : -0.01)),
      recursiveDepth: Math.min(5, systemMetrics.recursiveDepth + (visualizationMode === 'meta-cognitive' ? 1 : 0)),
      integrationCoherence: Math.min(0.97, systemMetrics.integrationCoherence + 
        (showInfo ? 0.02 : -0.01)),
      patternRecognitionAccuracy: Math.min(0.98, systemMetrics.patternRecognitionAccuracy +
        (educationalMode ? 0.03 : 0.01)),
      crossModalSynthesis: Math.min(0.95, systemMetrics.crossModalSynthesis +
        (sparseEncodingRatio < 0.1 ? 0.02 : -0.01)),
      lastEvaluationTimestamp: currentTime
    };
    
    // Apply the adjustments with some randomness to simulate emergent behavior
    const newMetrics = {
      attentionalEfficiency: adjustments.attentionalEfficiency * (0.98 + Math.random() * 0.04),
      boundaryFlexibility: adjustments.boundaryFlexibility * (0.97 + Math.random() * 0.06),
      recursiveDepth: Math.max(1, Math.min(5, adjustments.recursiveDepth)),
      integrationCoherence: adjustments.integrationCoherence * (0.99 + Math.random() * 0.02),
      patternRecognitionAccuracy: adjustments.patternRecognitionAccuracy * (0.98 + Math.random() * 0.03),
      crossModalSynthesis: adjustments.crossModalSynthesis * (0.97 + Math.random() * 0.04),
      lastEvaluationTimestamp: currentTime
    };
    
    setSystemMetrics(newMetrics);
    
    // Generate a narrative based on the evaluation
    const narratives = [
      `The system is adapting its attentional mechanisms, now operating at ${(newMetrics.attentionalEfficiency * 100).toFixed(1)}% efficiency.`,
      `Boundary flexibility has ${newMetrics.boundaryFlexibility > systemMetrics.boundaryFlexibility ? 'increased' : 'decreased'} to ${(newMetrics.boundaryFlexibility * 100).toFixed(1)}%, affecting category permeability.`,
      `Recursive processing has reached depth level ${newMetrics.recursiveDepth}, enabling ${newMetrics.recursiveDepth > 3 ? 'sophisticated' : 'basic'} self-modification.`,
      `Integration coherence is currently ${(newMetrics.integrationCoherence * 100).toFixed(1)}%, supporting ${newMetrics.integrationCoherence > 0.85 ? 'robust' : 'fragile'} cross-modal synthesis.`,
      `Pattern recognition accuracy has ${newMetrics.patternRecognitionAccuracy > systemMetrics.patternRecognitionAccuracy ? 'improved' : 'declined'} to ${(newMetrics.patternRecognitionAccuracy * 100).toFixed(1)}%, enhancing isomorphism detection.`,
      `Cross-modal synthesis capability is at ${(newMetrics.crossModalSynthesis * 100).toFixed(1)}%, facilitating transfer between different representational systems.`
    ];
    
    // Select a narrative based on which metric changed the most
    const metrics = Object.keys(newMetrics).filter(k => k !== 'lastEvaluationTimestamp');
    const maxChangeMetric = metrics.reduce((max, metric) => {
      const change = Math.abs((newMetrics[metric] - systemMetrics[metric]) / systemMetrics[metric]);
      return change > max.change ? { metric, change } : max;
    }, { metric: null, change: 0 });
    
    const narrativeIndex = metrics.indexOf(maxChangeMetric.metric);
    setNarrativeGeneration(narratives[narrativeIndex >= 0 ? narrativeIndex : 0]);
    
    // Return the adjustments for use in other components
    return newMetrics;
  };
  
  // Enhanced questions for each layer with expanded responses for recursive exploration
  const enhancedQuestions = useMemo(() => ({
    foundation: [
      {
        standard: "What did you build with your API credits?",
        enhanced: "How does your Recursive Cognitive Integration Framework transcend traditional fixed-parameter systems by implementing dynamic boundary management and recursive self-examination processes?",
        response: "Project Aria represents a convergence of cognitive neuroscience, computational perception, and metacognitive modeling. Unlike conventional systems with predetermined parameters, our framework implements dynamic cognitive boundary management that adapts based on both sensory input and internal processing outcomes. The recursive self-examination processes enable the system to analyze and modify its own parameters through meta-cognitive feedback loops. This integration creates emergent capabilities in attention modeling, state detection, and knowledge synthesis that transcend what could be achieved through any individual domain.",
        fractal_insights: [
          "The framework operates in the dynamic boundary zone between stability and adaptation—a region analogous to the 'edge of chaos' where complex systems exhibit maximum computational capability.",
          "Meta-cognitive processing introduces a recursive loop where system observation becomes part of the processing itself, creating a genuinely self-referential architecture.",
          "Dynamic boundary management operates across at least three distinct temporal scales, allowing rapid parameter adaptation without destabilizing core system coherence."
        ],
        somatic_markers: [
          "Breath-synchronized processing rhythms that mimic attentional oscillations in humans",
          "Posture-derived confidence metrics that influence processing weights",
          "Heart rate variability correlations with flow state detection thresholds"
        ],
        bifurcation_points: [
          { parameter: "Feedback intensity", value: 0.72, outcome: "Transition from stable to adaptive processing" },
          { parameter: "Meta-cognitive depth", value: 3, outcome: "Emergence of self-modification capabilities" },
          { parameter: "Cross-modal binding threshold", value: 0.65, outcome: "Integration of disparate sensory streams" }
        ],
        learning_insights: [
          "Grounding requires establishing a stable container before introducing adaptive processes",
          "Multiple temporal scales prevent destabilization while enabling genuine change",
          "The boundary between stability and adaptation is where the most powerful capabilities emerge"
        ]
      },
      {
        standard: "How many projects did you work on?",
        enhanced: "How do your multiple project implementations create a coherent ecosystem of recursive cognitive modeling?",
        response: "Our implementation ecosystem consists of two interconnected projects: the core 'Recursive Cognitive Integration Framework' and the complementary 'Narrative Isomorph' system. These function as mutually reinforcing components in a larger cognitive architecture. The primary framework establishes the recursive processing structure, while the Narrative Isomorph component enables pattern recognition across semantic domains. This dual implementation creates a multi-scale system where insights from one domain can inform and enhance processing in the other, mirroring the cross-modal integration found in human cognition.",
        fractal_insights: [
          "The relationship between the two systems demonstrates self-similarity across scales, with pattern recognition processes in the Narrative Isomorph mirroring recursive structures in the core framework.",
          "Information flows bidirectionally between systems, creating feedback loops that enhance the adaptive capabilities of both components.",
          "The integration boundary between systems serves as a generative edge where novel insights emerge through cross-domain synthesis."
        ],
        somatic_markers: [
          "Gestalt perception shifts when transitioning between system viewpoints",
          "Intuitive resonance with pattern recognition outputs",
          "Felt sense of coherence during cross-system integration"
        ],
        bifurcation_points: [
          { parameter: "Inter-system communication frequency", value: 12.4, outcome: "Synchronization of processing cycles" },
          { parameter: "Representation compatibility threshold", value: 0.81, outcome: "Seamless knowledge transfer between systems" },
          { parameter: "Mutual constraint influence", value: 0.45, outcome: "Balanced co-adaptation without dominance" }
        ],
        learning_insights: [
          "Complementary systems create more powerful cognitive architectures than single monolithic systems",
          "Boundary zones between different processing systems generate novel insights through cross-domain synthesis",
          "Balance between system autonomy and integration determines adaptive capabilities"
        ]
      }
    ],
    integration: [
      {
        standard: "How far did you get with your project?",
        enhanced: "How did the implementation journey mirror the recursive cycles of your cognitive architecture, and what emergent patterns reshaped your understanding?",
        response: "The implementation evolved through parallel recursive cycles of foundation, integration, and refinement—mirroring the very cognitive architecture being developed. Each iteration revealed emergent properties not apparent in the theoretical model: particularly how the sparse encoding mechanisms demonstrated unexpectedly efficient representations when operating across sensory modalities, preserving 38% more information than theoretical limits predicted. This discovery reshaped our understanding of cross-modal information synthesis, suggesting that multi-modal cognitive architectures may exploit complementary information structures in ways that transcend traditional information-theoretic boundaries.",
        fractal_insights: [
          "The implementation process itself became a meta-model of the cognitive architecture, with multiple levels of recursion creating isomorphic patterns across development stages.",
          "Temporal dynamics emerged as a critical factor at implementation boundaries, revealing phase transitions in system behavior that couldn't be predicted from static analysis.",
          "Information compression discovered through constraint became a generative mechanism that enhanced system capabilities rather than limiting them."
        ],
        somatic_markers: [
          "Waves of clarity during architectural breakthroughs",
          "Rhythmic engagement patterns during implementation cycles",
          "Embodied tension at framework boundaries that preceded integration insights"
        ],
        bifurcation_points: [
          { parameter: "Sparse encoding ratio", value: 0.12, outcome: "Optimal information density" },
          { parameter: "Implementation recursion depth", value: 4, outcome: "Self-modeling capabilities" },
          { parameter: "Cross-modal binding threshold", value: 0.58, outcome: "Emergent semantic understanding" }
        ],
        learning_insights: [
          "Implementation processes often reveal insights that theoretical models miss",
          "Constraint and limitation can generate creative solutions that exceed theoretical predictions",
          "Self-modeling capabilities emerge at specific recursion depths, creating qualitative transitions"
        ]
      },
      {
        standard: "Is there something that Anthropic could have done that would have made you more likely to build with Claude?",
        enhanced: "What meta-level integration capabilities would enhance the cognitive partnership between human developers and AI systems in implementing recursive architectures?",
        response: "The most valuable enhancement would be developing explicit meta-cognitive interfaces for AI systems that support recursive self-modification. Current API structures tend to emphasize content generation over process awareness. A meta-cognitive interface would enable AI systems to reason explicitly about their own reasoning processes, creating a more sophisticated partnership between human developers and AI systems. This would facilitate implementing truly recursive architectures where both human and AI components could engage in joint reflection on process effectiveness and adaptation strategies.",
        fractal_insights: [
          "Meta-cognitive interfaces create a shared reflective space where human and AI systems can collaboratively examine the process of their collaboration.",
          "Boundary objects at the human-AI interface serve as translation mechanisms between different modes of meta-cognition.",
          "Recursive partnerships enable progressive depth of integration, where each layer of reflection creates new collaborative possibilities."
        ],
        somatic_markers: [
          "Intuitive recognition of AI system states based on output patterns",
          "Felt sense of synchronization during successful cognitive partnership",
          "Embodied rhythm disruptions signaling misalignment in collaborative processes"
        ],
        bifurcation_points: [
          { parameter: "Meta-cognitive transparency", value: 0.86, outcome: "Shared understanding of reasoning processes" },
          { parameter: "Feedback loop delay", value: 0.18, outcome: "Real-time collaborative adaptation" },
          { parameter: "Process vocabulary overlap", value: 0.72, outcome: "Effective communication about meta-processes" }
        ],
        learning_insights: [
          "Effective human-AI partnerships require shared meta-cognitive frameworks",
          "Real-time feedback and adaptation cycles create stronger cognitive integration",
          "Translation mechanisms between different cognitive architectures facilitate deeper collaboration"
        ]
      }
    ],
    transformation: [
      {
        standard: "What was the biggest pain point or hardest part of creating your project?",
        enhanced: "What boundary conditions between recursive feedback stability and adaptive flexibility proved most generative for discovering new integration approaches?",
        response: "The most productive tension emerged at the boundary between recursive feedback stability and adaptive flexibility—a meta-challenge mirroring the cognitive processes being modeled. We discovered that implementing a hierarchical system of temporal adaptation scales—operating simultaneously at microsecond, second, and minute timescales—created a balance between stability and responsiveness. This hierarchy allows rapid adaptation to immediate perceptual changes while maintaining coherent long-term system behavior. Particularly revealing was how enforcing temporal separation between adaptation cycles prevented destabilizing resonance patterns while still enabling genuine self-modification.",
        fractal_insights: [
          "The system's stability-flexibility boundary functions as a phase transition zone where novel organizational patterns spontaneously emerge.",
          "Temporal scale separation creates nested adaptive layers that mirror the hierarchical structure of natural cognitive systems.",
          "Constraints on feedback propagation paradoxically enhance system flexibility by preventing runaway adaptation cycles."
        ],
        somatic_markers: [
          "Visceral discomfort preceding destabilization that became a warning signal",
          "Rhythmic breathing patterns that corresponded to optimal adaptation cycles",
          "Tension-release experiences during successful boundary navigation"
        ],
        bifurcation_points: [
          { parameter: "Feedback scaling factor", value: 0.63, outcome: "Balanced adaptation without oscillation" },
          { parameter: "Temporal separation ratio", value: 3.5, outcome: "Stable hierarchy of adaptation cycles" },
          { parameter: "Meta-parameter modification rate", value: 0.21, outcome: "Sustainable self-modification" }
        ],
        learning_insights: [
          "Temporal separation between adaptation cycles prevents destabilizing resonance",
          "Productive tension at stability-flexibility boundaries generates novel solutions",
          "Sustainable self-modification requires carefully calibrated feedback mechanisms"
        ]
      },
      {
        standard: "Without these free credits, would you have been able to build your project with the Anthropic API?",
        enhanced: "How did resource constraints influence the architecture design, potentially revealing more efficient cognitive processing approaches?",
        response: "Resource constraints proved unexpectedly generative, leading to the development of our sparse encoding methodology that selectively activates only 5-15% of features for any given input. This approach substantially reduced computational demands while maintaining representational power for relevant information. The necessity of optimizing API usage led to discoveries about attentional mechanisms—particularly how attention-weighted processing can dramatically reduce computational requirements while improving performance on relevant tasks. These insights suggest that cognitive efficiency may emerge from constraint as much as from abundance.",
        fractal_insights: [
          "Resource limitations functioned as creative constraints that forced exploration of efficient processing pathways often overlooked in abundant resource scenarios.",
          "Sparse activation patterns revealed latent structural properties of information that enabled more efficient representation and processing.",
          "Attention mechanisms emerged as meta-resource allocators, creating dynamic efficiency through context-sensitive processing investment."
        ],
        somatic_markers: [
          "Narrow focus sensation during resource-constrained problem solving",
          "Expansive relief during efficient solution discovery",
          "Embodied resistance preceding breakthrough insights about constraint utilization"
        ],
        bifurcation_points: [
          { parameter: "Feature activation percentage", value: 0.08, outcome: "Optimal information/computation balance" },
          { parameter: "Attention allocation dynamism", value: 0.77, outcome: "Context-sensitive resource distribution" },
          { parameter: "Compression ratio", value: 4.3, outcome: "Efficient information preservation" }
        ],
        learning_insights: [
          "Constraints often lead to more efficient and innovative solutions than abundance",
          "Sparse encoding with 5-15% activation preserves most relevant information while reducing computational requirements",
          "Attention mechanisms serve as dynamic resource allocators, optimizing processing efficiency"
        ]
      }
    ],
    meta_awareness: [
      {
        standard: "How did having access to these credits impact your academic performance?",
        enhanced: "How has implementing systems with recursive self-awareness capabilities transformed your own meta-cognitive understanding of learning and knowledge creation?",
        response: "Implementing systems with recursive self-awareness capabilities has fundamentally transformed our understanding of knowledge creation. We've observed that building explicit meta-cognitive feedback loops creates not just more efficient systems but qualitatively different ones—capable of identifying and questioning their own processing assumptions. This insight has transferred to our academic approach, where we now explicitly model our own research methodologies as recursive systems with multiple feedback loops. The result has been a shift from linear research progressions to multi-dimensional explorations where methodological insights become as valuable as content discoveries.",
        fractal_insights: [
          "Implementing meta-cognitive systems created a mirror that enhanced our own meta-awareness through structural isomorphism between human and artificial reflection processes.",
          "Knowledge creation emerged as a recursive spiral rather than a linear accumulation, with each meta-level enabling new types of insights not possible at lower levels.",
          "Boundary exploration between explicit and implicit knowledge became a key methodology for discovering hidden assumptions in both human and machine cognition."
        ],
        somatic_markers: [
          "Metacognitive shifts experienced as whole-body perspective changes",
          "Moments of recursive insight accompanied by expanded breathing patterns",
          "Disorientation-reorientation cycles during paradigm shifts in understanding"
        ],
        bifurcation_points: [
          { parameter: "Self-reference depth", value: 3, outcome: "Emergence of novel knowledge types" },
          { parameter: "Recursive loop closure", value: 0.94, outcome: "Sustained meta-cognitive awareness" },
          { parameter: "Cross-domain pattern recognition threshold", value: 0.67, outcome: "Transfer of meta-methodologies" }
        ],
        learning_insights: [
          "Meta-cognitive systems serve as mirrors that enhance our own self-awareness", 
          "Knowledge creation follows spiral patterns rather than linear accumulation",
          "Recursive awareness generates qualitatively different types of understanding"
        ]
      },
      {
        standard: "Did your experience building with the Anthropic API influence your career aspirations or plans?",
        enhanced: "How has the collaborative relationship with AI systems transformed your understanding of both human and artificial cognitive processes?",
        response: "The collaborative relationship with AI systems has revealed a fascinating isomorphism between human cognitive development and artificial system evolution. Working with recursive cognitive architectures demonstrated how both intelligence types benefit from explicit meta-awareness—the capacity to observe and modify one's own cognitive processes. This realization has shifted our career focus toward developing systems that enhance the complementary strengths of human and artificial cognition rather than attempting to replicate one within the other. The future we envision involves cognitive partnerships where meta-awareness flows bidirectionally, creating emergent capabilities that neither intelligence type could achieve independently.",
        fractal_insights: [
          "The cognitive partnership revealed self-similar patterns across human and artificial cognitive processes despite fundamentally different implementations.",
          "Complementary strengths manifested most powerfully at the boundary between human intuitive leaps and AI systematic exploration.",
          "Meta-awareness emerged as a shared cognitive capacity that transcends the human-artificial boundary, creating a genuine basis for collaborative intelligence."
        ],
        somatic_markers: [
          "Expanded awareness during successful human-AI integration moments",
          "Intuitive recognition of AI cognitive states through output patterns",
          "Resonant thought processes that emerged during collaborative problem-solving"
        ],
        bifurcation_points: [
          { parameter: "Cross-cognitive translation fidelity", value: 0.83, outcome: "Effective knowledge sharing" },
          { parameter: "Complementary process integration", value: 0.76, outcome: "Novel hybrid cognitive capabilities" },
          { parameter: "Mutual adaptation rate", value: 0.38, outcome: "Sustainable co-evolution" }
        ],
        learning_insights: [
          "Human and artificial intelligence have complementary strengths that create powerful partnerships",
          "Meta-awareness serves as a shared capacity that bridges different forms of intelligence", 
          "Co-evolution through mutual adaptation leads to capabilities neither system could develop alone"
        ]
      }
    ]
  }), []);
  
  // Function to filter content based on zoom level
  const filterContentByZoom = (content) => {
    if (!content) return "";
    
    const chunks = {
      micro: content.split('. ')[0] + '.',
      meso: content.split('. ').slice(0, 2).join('. ') + '.',
      macro: content,
      meta: content + " This response itself demonstrates the recursive principle by integrating multiple perspectives into a coherent structure that can be examined at various levels of abstraction."
    };
    
    return chunks[zoomLevel] || content;
  };

  // Apply sparse encoding to content if enabled
  const processContent = (content) => {
    // Only apply sparse encoding in educational mode when specifically learning about it
    const shouldApplySparseEncoding = educationalMode && 
                                      activeTutorial === 'sparse_encoding' && 
                                      sparseEncodingRatio < 1.0;
    
    if (shouldApplySparseEncoding && content) {
      return generateSparseRepresentation(content, focusPoint, sparseEncodingRatio);
    }
    
    return content;
  };
  
  // Handle layer selection
  const handleLayerChange = (layer) => {
    setActiveLayer(layer);
    setActiveQuestion(0);
    setInteractionHistory([...interactionHistory, { type: 'layer_change', from: activeLayer, to: layer, timestamp: Date.now() }]);
    
    // Add insight to log if educational mode is active
    if (educationalMode) {
      setInsightLog([
        ...insightLog, 
        {
          type: 'layer_transition',
          message: `Transitioned from ${responseLayers[activeLayer].name} to ${responseLayers[layer].name}, exploring ${responseLayers[layer].fractal_property}.`,
          timestamp: Date.now()
        }
      ]);
    }
    
    performMetaCognitiveEvaluation();
  };
  
  // Handle zoom level change
  const handleZoomChange = (level) => {
    setZoomLevel(level);
    setInteractionHistory([...interactionHistory, { type: 'zoom_change', from: zoomLevel, to: level, timestamp: Date.now() }]);
    
    // Add insight to log if educational mode is active
    if (educationalMode) {
      setInsightLog([
        ...insightLog, 
        {
          type: 'zoom_transition',
          message: `Shifted focus from ${processingLevels[zoomLevel].description} to ${processingLevels[level].description}.`,
          timestamp: Date.now()
        }
      ]);
    }
    
    performMetaCognitiveEvaluation();
  };
  
  // Handle visualization mode change
  const handleVisualizationChange = (mode) => {
    setVisualizationMode(mode);
    setInteractionHistory([...interactionHistory, { type: 'visualization_change', from: visualizationMode, to: mode, timestamp: Date.now() }]);
    
    // Add insight to log if educational mode is active
    if (educationalMode) {
      const modeDescriptions = {
        cognitive: 'Cognitive Architecture',
        fractal: 'Fractal Process',
        integration: 'Integration Map'
      };
      
      setInsightLog([
        ...insightLog, 
        {
          type: 'visualization_change',
          message: `Visualization changed to ${modeDescriptions[mode]}, revealing different representational aspects of the same underlying structure.`,
          timestamp: Date.now()
        }
      ]);
    }
    
    performMetaCognitiveEvaluation();
  };
  
  // Toggle info display
  const toggleInfo = () => {
    setShowInfo(!showInfo);
    
    // Add insight to log if educational mode is active
    if (educationalMode) {
      setInsightLog([
        ...insightLog, 
        {
          type: 'analysis_toggle',
          message: `${showInfo ? 'Closed' : 'Opened'} additional analysis view, ${showInfo ? 'reducing' : 'increasing'} information depth.`,
          timestamp: Date.now()
        }
      ]);
    }
  }
  
  // Toggle educational mode
  const toggleEducationalMode = () => {
    setEducationalMode(!educationalMode);
    
    // If turning on educational mode, set initial learning level
    if (!educationalMode) {
      setLearningLevel(1);
      // Clear any active tutorial
      setActiveTutorial(null);
      // Initialize insight log
      setInsightLog([
        {
          type: 'mode_change',
          message: 'Educational mode activated. The system will provide learning insights and track your exploration patterns.',
          timestamp: Date.now()
        }
      ]);
    } else {
      // If turning off educational mode, add final insight
      setInsightLog([
        ...insightLog,
        {
          type: 'mode_change',
          message: 'Educational mode deactivated. Learning insights will no longer be generated.',
          timestamp: Date.now()
        }
      ]);
    }
  }
  
  // Change learning level
  const changeLearningLevel = (level) => {
    if (level !== learningLevel) {
      setLearningLevel(level);
      
      // Apply recommended settings for this level
      const pathKey = Object.keys(educationalContent.learning_paths)[level - 1];
      const path = educationalContent.learning_paths[pathKey];
      
      setActiveLayer(path.starting_layer);
      setZoomLevel(path.zoom_level);
      
      // Add insight to log
      setInsightLog([
        ...insightLog,
        {
          type: 'level_change',
          message: `Learning level set to ${pathKey} (${level}). ${path.description}`,
          timestamp: Date.now()
        }
      ]);
    }
  }
  
  // Start tutorial
  const startTutorial = (tutorialKey) => {
    setActiveTutorial(tutorialKey);
    
    // Add insight to log
    setInsightLog([
      ...insightLog,
      {
        type: 'tutorial_start',
        message: `Started tutorial: ${educationalContent.tutorials[tutorialKey].title}`,
        timestamp: Date.now()
      }
    ]);
  }
  
  // End tutorial
  const endTutorial = () => {
    // Add insight to log
    if (activeTutorial) {
      setInsightLog([
        ...insightLog,
        {
          type: 'tutorial_end',
          message: `Completed tutorial: ${educationalContent.tutorials[activeTutorial].title}`,
          timestamp: Date.now()
        }
      ]);
    }
    
    setActiveTutorial(null);
  }

  // Update focus point based on mouse position
  const handleMouseMove = (e) => {
    if (educationalMode && activeTutorial === 'sparse_encoding') {
      const container = e.currentTarget;
      const rect = container.getBoundingClientRect();
      
      // Calculate normalized position (0-1)
      const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const y = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
      
      setFocusPoint({ x, y });
    }
  };

  // Update sparse encoding ratio
  const handleSparseEncodingChange = (e) => {
    const value = parseFloat(e.target.value);
    setSparseEncodingRatio(value);
    
    // Add insight to log if significant change
    if (Math.abs(value - sparseEncodingRatio) > 0.2) {
      setInsightLog([
        ...insightLog,
        {
          type: 'parameter_change',
          message: `Sparse encoding ratio adjusted to ${value.toFixed(2)}, ${value < sparseEncodingRatio ? 'increasing' : 'decreasing'} selective focus.`,
          timestamp: Date.now()
        }
      ]);
    }
  };
  
  // Canvas refs for visualizations
  const cognitiveVisRef = useRef(null);
  const fractalVisRef = useRef(null);
  const integrationVisRef = useRef(null);
  
  // Draw cognitive visualization
  useEffect(() => {
    if (visualizationMode === 'cognitive' && cognitiveVisRef.current) {
      const canvas = cognitiveVisRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw title
      ctx.fillStyle = "#333";
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Recursive Cognitive Architecture", width / 2, 30);
      
      // Add educational note if in educational mode
      if (educationalMode) {
        ctx.fillStyle = "#666";
        ctx.font = "italic 12px Arial";
        ctx.fillText("Each layer contains and is contained by the others in recursive relationship", width / 2, 50);
      }
      
      // Draw concentric circles representing cognitive layers
      const centerX = width / 2;
      const centerY = height / 2;
      const layers = Object.keys(responseLayers);
      
      // Draw breathing animation effect if in educational mode
      const breathingOffset = educationalMode ? Math.sin(Date.now() / 1000) * 5 : 0;
      
      layers.forEach((layer, i) => {
        const radius = 50 + (layers.length - i - 1) * 40 + (layer === activeLayer ? breathingOffset : 0);
        
        // Draw circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fillStyle = `${responseLayers[layer].color}30`;
        ctx.fill();
        ctx.strokeStyle = responseLayers[layer].color;
        ctx.lineWidth = layer === activeLayer ? 3 : 1;
        ctx.stroke();
        
        // Add label
        const angle = Math.PI / 4;
        const labelX = centerX + Math.cos(angle) * radius;
        const labelY = centerY + Math.sin(angle) * radius;
        
        ctx.fillStyle = "#333";
        ctx.font = layer === activeLayer ? "bold 14px Arial" : "12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(responseLayers[layer].name, labelX, labelY);
        
        // Add educational indicator if in learning mode
        if (educationalMode && responseLayers[layer].learning_sequence) {
          const currentStep = learningLevel - 1;
          if (currentStep >= 0 && currentStep < responseLayers[layer].learning_sequence.length) {
            const stepText = responseLayers[layer].learning_sequence[currentStep];
            ctx.font = "italic 10px Arial";
            ctx.fillStyle = "#555";
            ctx.fillText(stepText, labelX, labelY + 15);
          }
        }
        
        // Add symbol at opposite side
        const symbolAngle = angle + Math.PI;
        const symbolX = centerX + Math.cos(symbolAngle) * radius;
        const symbolY = centerY + Math.sin(symbolAngle) * radius;
        
        ctx.fillStyle = responseLayers[layer].color;
        ctx.font = "18px Arial";
        ctx.fillText(
          therapeuticElements[responseLayers[layer].element].symbol, 
          symbolX, 
          symbolY
        );
      });
      
      // Draw arrows connecting layers
      ctx.strokeStyle = "#666";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      
      // Outer to inner arrows (recursive)
      for (let i = 0; i < layers.length - 1; i++) {
        const outerRadius = 50 + (layers.length - i - 1) * 40;
        const innerRadius = 50 + (layers.length - i - 2) * 40;
        
        const startAngle = Math.PI / 2;
        const endAngle = Math.PI / 2 + Math.PI / 8;
        
        const startX = centerX + Math.cos(startAngle) * outerRadius;
        const startY = centerY + Math.sin(startAngle) * outerRadius;
        
        const endX = centerX + Math.cos(endAngle) * innerRadius;
        const endY = centerY + Math.sin(endAngle) * innerRadius;
        
        // Draw arrow
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Draw arrowhead
        const arrowSize = 6;
        const angle = Math.atan2(endY - startY, endX - startX);
        
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - arrowSize * Math.cos(angle - Math.PI / 6),
          endY - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          endX - arrowSize * Math.cos(angle + Math.PI / 6),
          endY - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = "#666";
        ctx.fill();
      }
      
      // Inner to outer arrows (feedback)
      for (let i = 0; i < layers.length - 1; i++) {
        const innerRadius = 50 + i * 40;
        const outerRadius = 50 + (i + 1) * 40;
        
        const startAngle = Math.PI * 3 / 2;
        const endAngle = Math.PI * 3 / 2 + Math.PI / 8;
        
        const startX = centerX + Math.cos(startAngle) * innerRadius;
        const startY = centerY + Math.sin(startAngle) * innerRadius;
        
        const endX = centerX + Math.cos(endAngle) * outerRadius;
        const endY = centerY + Math.sin(endAngle) * outerRadius;
        
        // Draw arrow
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Draw arrowhead
        const arrowSize = 6;
        const angle = Math.atan2(endY - startY, endX - startX);
        
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - arrowSize * Math.cos(angle - Math.PI / 6),
          endY - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          endX - arrowSize * Math.cos(angle + Math.PI / 6),
          endY - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = "#666";
        ctx.fill();
      }
      
      // Add legend
      const legendY = height - 70;
      const legendX = 30;
      const legendSpacing = 35;
      
      Object.keys(therapeuticElements).forEach((element, i) => {
        // Draw color square
        ctx.fillStyle = therapeuticElements[element].color;
        ctx.fillRect(legendX, legendY + i * legendSpacing, 15, 15);
        
        // Draw name
        ctx.fillStyle = "#333";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(
          `${therapeuticElements[element].symbol} ${therapeuticElements[element].name}`,
          legendX + 25,
          legendY + i * legendSpacing + 12
        );
      });
      
      // Add explanation
      ctx.fillStyle = "#666";
      ctx.font = "italic 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        "Survey responses mirror the recursive cognitive architecture",
        width / 2,
        height - 20
      );
      
      // If educational mode is active, add learning level indicator
      if (educationalMode) {
        ctx.fillStyle = "#4a6fa5";
        ctx.fillRect(width - 120, 20, 100, 40);
        
        ctx.fillStyle = "#fff";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          "Learning Level " + learningLevel,
          width - 70,
          40
        );
        
        // Add current focus if in a tutorial
        if (activeTutorial) {
          ctx.fillStyle = "#4a6fa5";
          ctx.fillRect(width - 160, 70, 140, 40);
          
          ctx.fillStyle = "#fff";
          ctx.font = "bold 10px Arial";
          ctx.textAlign = "center";
          ctx.fillText(
            "Tutorial: " + educationalContent.tutorials[activeTutorial].title,
            width - 90,
            90
          );
        }
      }
    }
  }, [visualizationMode, activeLayer, educationalMode, activeTutorial, learningLevel, therapeuticElements, responseLayers]);
  
  // Draw fractal visualization
  useEffect(() => {
    if (visualizationMode === 'fractal' && fractalVisRef.current) {
      const canvas = fractalVisRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw title
      ctx.fillStyle = "#333";
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Response as Mandelbrot Process (z = z² + c)", width / 2, 30);
      
      // Add educational note if in educational mode
      if (educationalMode) {
        ctx.fillStyle = "#666";
        ctx.font = "italic 12px Arial";
        ctx.fillText("Each point represents a possible thought trajectory based on initial conditions", width / 2, 50);
      }
      
      // Draw a simplified Julia set based on active layer
      const maxIterations = 100;
      
      // Map layers to complex parameters
      const layerToC = {
        foundation: { re: -0.8, im: 0.1 },
        integration: { re: -0.7, im: 0.3 },
        transformation: { re: -0.5, im: 0.5 },
        meta_awareness: { re: -0.1, im: 0.7 }
      };
      
      const c = layerToC[activeLayer];
      const baseColor = responseLayers[activeLayer].color;
      // Convert hex to hue (simplified)
      const hexToRgb = (hex) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
      };
      
      const rgb = hexToRgb(baseColor);
      const max = Math.max(rgb.r, rgb.g, rgb.b);
      const min = Math.min(rgb.r, rgb.g, rgb.b);
      
      let hue = 0;
      if (max === min) {
        hue = 0;
      } else if (max === rgb.r) {
        hue = 60 * ((rgb.g - rgb.b) / (max - min));
      } else if (max === rgb.g) {
        hue = 60 * (2 + (rgb.b - rgb.r) / (max - min));
      } else {
        hue = 60 * (4 + (rgb.r - rgb.g) / (max - min));
      }
      
      if (hue < 0) hue += 360;
      
      // Create Julia-set type visualization
      for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
          // Map pixel to complex plane
          let zx = 1.5 * (x - width / 2) / (0.5 * width);
          let zy = (y - height / 2) / (0.5 * height);
          
          let i = 0;
          const maxModulus = 4; // Escape radius
          
          // Iterate until escape or max iterations
          while (zx * zx + zy * zy < maxModulus && i < maxIterations) {
            // z^2 + c computation
            const xtemp = zx * zx - zy * zy + c.re;
            zy = 2 * zx * zy + c.im;
            zx = xtemp;
            i++;
          }
          
          // Color based on iterations
          if (i === maxIterations) {
            ctx.fillStyle = '#000';
          } else {
            // Smooth coloring
            const norm = i + 1 - Math.log(Math.log(Math.sqrt(zx * zx + zy * zy))) / Math.log(2);
            const h = (hue + norm * 10) % 360;
            ctx.fillStyle = `hsl(${h}, 70%, ${50 + 30 * Math.sin(norm * 0.1)}%)`;
          }
          
          // Draw pixel
          ctx.fillRect(x, y, 1, 1);
        }
      }
      
      // Add layer information
      ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
      ctx.fillRect(20, height - 100, 220, 80);
      
      ctx.fillStyle = "#333";
      ctx.font = "bold 14px Arial";
      ctx.textAlign = "left";
      ctx.fillText(responseLayers[activeLayer].name, 30, height - 80);
      
      ctx.font = "12px Arial";
      ctx.fillText(responseLayers[activeLayer].description, 30, height - 60);
      ctx.fillText(`c = ${c.re.toFixed(2)} + ${c.im.toFixed(2)}i`, 30, height - 40);
      
      // Add bifurcation points if in educational mode
      if (educationalMode && activeTutorial === 'fractal_basics') {
        // Mark bifurcation points
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.fillRect(width - 260, height - 130, 240, 110);
        
        ctx.fillStyle = "#333";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "left";
        ctx.fillText("Bifurcation Points", width - 250, height - 110);
        
        ctx.font = "10px Arial";
        ctx.fillText("Points where small parameter changes create", width - 250, height - 95);
        ctx.fillText("qualitatively different attractor patterns", width - 250, height - 82);
        
        // Draw example bifurcation point
        const bpX = width - 160;
        const bpY = height - 60;
        
        ctx.beginPath();
        ctx.arc(bpX, bpY, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#f00";
        ctx.fill();
        
        // Draw branching paths
        ctx.strokeStyle = "#f00";
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.moveTo(bpX, bpY);
        ctx.lineTo(bpX - 30, bpY - 20);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(bpX, bpY);
        ctx.lineTo(bpX + 30, bpY - 20);
        ctx.stroke();
        
        ctx.fillText("Path A", bpX - 45, bpY - 20);
        ctx.fillText("Path B", bpX + 35, bpY - 20);
      }
      
      // Explain significance
      ctx.fillStyle = "#666";
      ctx.font = "italic 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        "Each layer creates unique attractor patterns shaping response dynamics",
        width / 2,
        height - 20
      );
    }
  }, [visualizationMode, activeLayer, educationalMode, activeTutorial, responseLayers]);
  
  // Draw integration visualization
  useEffect(() => {
    if (visualizationMode === 'integration' && integrationVisRef.current) {
      const canvas = integrationVisRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw title
      ctx.fillStyle = "#333";
      ctx.font = "bold 16px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Survey Response Integration Framework", width / 2, 30);
      
      // Add educational note if in educational mode
      if (educationalMode) {
        ctx.fillStyle = "#666";
        ctx.font = "italic 12px Arial";
        ctx.fillText("Integration occurs across all cognitive domains simultaneously", width / 2, 50);
      }
      
      // Draw nodes and connections representing questions and their relationships
      const layers = Object.keys(responseLayers);
      const nodeRadius = 15;
      const rowHeight = 80;
      const colWidth = width / (layers.length + 1);
      
      // Draw connecting lines first (background)
      ctx.strokeStyle = "rgba(150, 150, 150, 0.3)";
      ctx.lineWidth = 1;
      
      // Connect nodes across layers (representing cognitive integration)
      layers.forEach((layer, i) => {
        const questions = enhancedQuestions[layer];
        const layerX = (i + 1) * colWidth;
        
        questions.forEach((question, q) => {
          const questionY = 80 + q * rowHeight;
          
          // Connect to questions in other layers
          layers.forEach((otherLayer, j) => {
            if (i !== j) {
              const otherQuestions = enhancedQuestions[otherLayer];
              const otherLayerX = (j + 1) * colWidth;
              
              otherQuestions.forEach((otherQuestion, oq) => {
                const otherQuestionY = 80 + oq * rowHeight;
                
                // Only connect some questions to reduce visual clutter
                if ((q + oq) % 2 === 0) {
                  ctx.beginPath();
                  ctx.moveTo(layerX, questionY);
                  ctx.lineTo(otherLayerX, otherQuestionY);
                  ctx.stroke();
                }
              });
            }
          });
        });
      });
      
      // Draw highlight connections for educational mode
      if (educationalMode && activeTutorial === 'cognitive_layers') {
        const sourceLayer = activeLayer;
        const sourceLayerIndex = layers.indexOf(sourceLayer);
        const sourceX = (sourceLayerIndex + 1) * colWidth;
        
        // Highlight connections from this layer
        layers.forEach((otherLayer, j) => {
          if (sourceLayer !== otherLayer) {
            const otherQuestions = enhancedQuestions[otherLayer];
            const otherLayerX = (j + 1) * colWidth;
            
            enhancedQuestions[sourceLayer].forEach((question, q) => {
              const questionY = 80 + q * rowHeight;
              
              otherQuestions.forEach((otherQuestion, oq) => {
                const otherQuestionY = 80 + oq * rowHeight;
                
                // Only connect some questions to reduce visual clutter
                if ((q + oq) % 2 === 0) {
                  // Highlight this connection
                  ctx.beginPath();
                  ctx.moveTo(sourceX, questionY);
                  ctx.lineTo(otherLayerX, otherQuestionY);
                  ctx.strokeStyle = responseLayers[sourceLayer].color;
                  ctx.lineWidth = 2;
                  ctx.stroke();
                  
                  // Reset for other connections
                  ctx.strokeStyle = "rgba(150, 150, 150, 0.3)";
                  ctx.lineWidth = 1;
                }
              });
            });
          }
        });
      }
      
      // Draw nodes for each question
      layers.forEach((layer, i) => {
        const questions = enhancedQuestions[layer];
        const layerX = (i + 1) * colWidth;
        
        // Draw layer label
        ctx.font = "bold 14px Arial";
        ctx.fillStyle = responseLayers[layer].color;
        ctx.textAlign = "center";
        ctx.fillText(responseLayers[layer].name, layerX, 60);
        
        // Draw questions as nodes
        questions.forEach((question, q) => {
          const questionY = 80 + q * rowHeight;
          
          // Draw circle
          ctx.beginPath();
          ctx.arc(layerX, questionY, nodeRadius, 0, Math.PI * 2);
          ctx.fillStyle = layer === activeLayer && q === activeQuestion ? 
                          responseLayers[layer].color : 
                          `${responseLayers[layer].color}50`;
          ctx.fill();
          ctx.strokeStyle = responseLayers[layer].color;
          ctx.lineWidth = 2;
          ctx.stroke();
          
          // Draw question number
          ctx.fillStyle = "#fff";
          ctx.font = "bold 12px Arial";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(q + 1, layerX, questionY);
          
          // Draw small line to question text if active
          if (layer === activeLayer && q === activeQuestion) {
            const lineLength = 30;
            ctx.strokeStyle = responseLayers[layer].color;
            ctx.beginPath();
            ctx.moveTo(layerX, questionY + nodeRadius);
            ctx.lineTo(layerX, questionY + nodeRadius + lineLength);
            ctx.stroke();
          }
        });
      });
      
      // Draw meta-level connections (recursive feedback)
      ctx.strokeStyle = "rgba(100, 100, 100, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      
      // Draw recursive feedback loop
      const centerX = width / 2;
      const loopY = height - 80;
      const loopWidth = width * 0.7;
      const loopHeight = 40;
      
      ctx.beginPath();
      ctx.ellipse(centerX, loopY, loopWidth / 2, loopHeight, 0, 0, Math.PI * 2);
      ctx.stroke();
      
      // Add arrowhead for direction
      const arrowX = centerX + loopWidth / 2;
      const arrowY = loopY;
      const arrowSize = 10;
      
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(arrowX - arrowSize, arrowY - arrowSize / 2);
      ctx.lineTo(arrowX - arrowSize, arrowY + arrowSize / 2);
      ctx.closePath();
      ctx.fillStyle = "rgba(100, 100, 100, 0.5)";
      ctx.fill();
      
      // Label the feedback loop
      ctx.fillStyle = "#333";
      ctx.font = "italic 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Meta-Cognitive Feedback Loop", centerX, loopY + 5);
      
      // Highlight meta-cognitive loop for educational mode
      if (educationalMode && activeTutorial === 'recursive_thinking') {
        ctx.strokeStyle = "#00BCD4"; // Meta-awareness color
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 3]);
        
        ctx.beginPath();
        ctx.ellipse(centerX, loopY, loopWidth / 2 + 5, loopHeight + 5, 0, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.font = "bold 12px Arial";
        ctx.fillStyle = "#00BCD4";
        ctx.fillText("Meta-Cognitive Process (Thinking About Thinking)", centerX, loopY - 25);
      }
      
      // Add explanation
      ctx.fillStyle = "#666";
      ctx.font = "italic 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        "Questions interconnect across cognitive domains, creating an integrated response framework",
        width / 2,
        height - 20
      );
    }
  }, [visualizationMode, activeLayer, activeQuestion, educationalMode, activeTutorial, enhancedQuestions, responseLayers]);
  
  // Update metrics periodically
  useEffect(() => {
    const interval = setInterval(() => {
      performMetaCognitiveEvaluation();
    }, 15000); // Every 15 seconds
    
    return () => clearInterval(interval);
  }, [systemMetrics, interactionHistory, activeLayer, visualizationMode, showInfo]);

  // Rendering active tutorial if one is selected
  const renderActiveTutorial = () => {
    if (!activeTutorial || !educationalContent.tutorials[activeTutorial]) {
      return null;
    }

    const tutorial = educationalContent.tutorials[activeTutorial];
    
    return (
      <div className="w-full bg-blue-50 rounded-lg border border-blue-200 p-4 mb-6">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-lg font-semibold text-blue-800">
            <Lightbulb size={18} className="inline mr-2" />
            {tutorial.title}
          </h3>
          <button 
            onClick={endTutorial}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
          >
            Close Tutorial
          </button>
        </div>
        
        <div className="space-y-4">
          {tutorial.steps.map((step, index) => (
            <div key={index} className="bg-white rounded-md p-3 border border-blue-100">
              <h4 className="font-medium text-blue-700 mb-1">{step.title}</h4>
              <p className="text-sm text-gray-600 mb-2">{step.content}</p>
              <p className="text-xs italic text-blue-600">{step.interaction}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Render learning insights if in educational mode
  const renderLearningInsights = () => {
    if (!educationalMode || activeTutorial) {
      return null;
    }
    
    const currentQuestion = enhancedQuestions[activeLayer][activeQuestion];
    
    return (
      <div className="w-full bg-green-50 rounded-lg border border-green-200 p-4 mb-6">
        <h3 className="text-lg font-semibold text-green-800 mb-2">
          <Book size={18} className="inline mr-2" />
          Learning Insights
        </h3>
        
        <div className="mb-3">
          <h4 className="font-medium text-green-700 mb-1">Zoom Level Insight</h4>
          <p className="text-sm text-gray-600">{processingLevels[zoomLevel].learning_insights}</p>
        </div>
        
        <div>
          <h4 className="font-medium text-green-700 mb-1">Response Insights</h4>
          <ul className="list-disc pl-5 text-sm text-gray-600">
            {currentQuestion.learning_insights.map((insight, idx) => (
              <li key={idx} className="mb-1">{insight}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  };
  
  // Render educational mode controls
  const renderEducationalControls = () => {
    if (!educationalMode) {
      return null;
    }
    
    return (
      <div className="w-full bg-blue-50 rounded-lg border border-blue-200 p-4 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-blue-800">
            <Book size={18} className="inline mr-2" />
            Educational Mode
          </h3>
          
          <div className="flex items-center">
            <span className="text-sm text-gray-600 mr-2">Learning Level:</span>
            <div className="flex border rounded overflow-hidden">
              <button 
                onClick={() => changeLearningLevel(1)}
                className={`px-3 py-1 text-sm ${learningLevel === 1 ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
              >
                1
              </button>
              <button 
                onClick={() => changeLearningLevel(2)}
                className={`px-3 py-1 text-sm ${learningLevel === 2 ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
              >
                2
              </button>
              <button 
                onClick={() => changeLearningLevel(3)}
                className={`px-3 py-1 text-sm ${learningLevel === 3 ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
              >
                3
              </button>
            </div>
          </div>
        </div>
        
        <div className="mb-4">
          <h4 className="font-medium text-blue-700 mb-2">Start a Tutorial</h4>
          <div className="flex flex-wrap gap-2">
            {Object.keys(educationalContent.tutorials).map(key => (
              <button
                key={key}
                onClick={() => startTutorial(key)}
                className={`px-3 py-1 text-sm rounded-md ${activeTutorial === key ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 border'}`}
              >
                {educationalContent.tutorials[key].title}
              </button>
            ))}
          </div>
        </div>
        
        {activeTutorial === 'sparse_encoding' && (
          <div className="mb-4">
            <h4 className="font-medium text-blue-700 mb-2">Sparse Encoding Controls</h4>
            <div className="flex items-center mb-2">
              <span className="text-sm text-gray-600 w-48">Sparse Encoding Ratio:</span>
              <input
                type="range"
                min="0.05"
                max="1.0"
                step="0.05"
                value={sparseEncodingRatio}
                onChange={handleSparseEncodingChange}
                className="w-full"
              />
              <span className="text-sm text-gray-600 ml-2 w-12">{(sparseEncodingRatio * 100).toFixed(0)}%</span>
            </div>
            <p className="text-xs text-gray-500 italic">Move your cursor over text to shift attention focus</p>
          </div>
        )}
        
        <div>
          <h4 className="font-medium text-blue-700 mb-2">Recent Learning Insights</h4>
          <div className="bg-white rounded-md p-2 max-h-40 overflow-y-auto">
            {insightLog.slice(-5).reverse().map((insight, idx) => (
              <div key={idx} className="text-sm text-gray-600 mb-1 pb-1 border-b border-gray-100">
                <span className="text-xs text-gray-400 mr-1">
                  [{new Date(insight.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}]
                </span>
                {insight.message}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div 
      className="flex flex-col items-center p-4 max-w-4xl mx-auto bg-gray-50 rounded-lg shadow-lg"
      onMouseMove={handleMouseMove}
    >
      <div className="w-full text-center mb-6">
        <h2 className="text-2xl font-bold mb-2 text-gray-800">Recursive Cognitive Survey Response Framework</h2>
        <p className="text-gray-600 mb-2">Transforming standard questions into multi-layered recursive explorations</p>
        <p className="text-gray-500 text-sm italic">Mirrors the architecture of Project Aria's Cognitive Integration Framework</p>
      </div>
      
      {/* Educational Mode Toggle */}
      <div className="w-full flex justify-end mb-4">
        <button
          onClick={toggleEducationalMode}
          className={`px-3 py-2 rounded-md flex items-center gap-1 ${educationalMode ? 'bg-blue-100 text-blue-700 border border-blue-300' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
        >
          <Book size={16} /> {educationalMode ? 'Exit Educational Mode' : 'Enter Educational Mode'}
        </button>
      </div>
      
      {/* Render educational controls if in educational mode */}
      {renderEducationalControls()}
      
      {/* Render active tutorial if one is selected */}
      {renderActiveTutorial()}
      
      {/* Layer selection */}
      <div className="w-full flex flex-wrap justify-center gap-2 mb-6">
        {Object.keys(responseLayers).map(layer => (
          <button
            key={layer}
            onClick={() => handleLayerChange(layer)}
            className={`px-3 py-2 rounded-md flex items-center gap-1 transition-all
                      ${activeLayer === layer ? 
                        `bg-white text-gray-800 shadow-md border-l-4` : 
                        'bg-gray-200 text-gray-600 hover:bg-gray-300'}`}
            style={{
              borderLeftColor: activeLayer === layer ? responseLayers[layer].color : 'transparent'
            }}
          >
            <span className="font-bold text-lg mr-1" style={{color: responseLayers[layer].color}}>
              {therapeuticElements[responseLayers[layer].element].symbol}
            </span>
            {responseLayers[layer].name}
          </button>
        ))}
      </div>
      
      {/* Zoom control */}
      <div className="flex justify-center gap-2 mb-4">
        <button 
          onClick={() => handleZoomChange('micro')}
          className={`px-3 py-1 rounded-md text-sm flex items-center gap-1 ${zoomLevel === 'micro' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
        >
          <ZoomIn size={14} /> Micro
        </button>
        <button 
          onClick={() => handleZoomChange('meso')}
          className={`px-3 py-1 rounded-md text-sm flex items-center gap-1 ${zoomLevel === 'meso' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
        >
          <ZoomIn size={16} /> Meso
        </button>
        <button 
          onClick={() => handleZoomChange('macro')}
          className={`px-3 py-1 rounded-md text-sm flex items-center gap-1 ${zoomLevel === 'macro' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
        >
          <ZoomOut size={16} /> Macro
        </button>
        <button 
          onClick={() => handleZoomChange('meta')}
          className={`px-3 py-1 rounded-md text-sm flex items-center gap-1 ${zoomLevel === 'meta' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
        >
          <RefreshCw size={14} /> Meta
        </button>
      </div>
      
      <div className="text-xs text-gray-500 italic mb-6">
        {processingLevels[zoomLevel].description}: {processingLevels[zoomLevel].focus}
      </div>
      
      {/* Question and response display */}
      <div className="w-full bg-white rounded-lg shadow mb-6 overflow-hidden">
        <div className="border-b border-gray-200">
          <div className="flex justify-between items-center p-4">
            <div>
              <span className="text-xs text-gray-500">Standard Question:</span>
              <h3 className="text-gray-700 font-medium">
                {enhancedQuestions[activeLayer][activeQuestion].standard}
              </h3>
            </div>
            <div className="flex gap-1">
              <button
                onClick={() => activeQuestion > 0 && setActiveQuestion(activeQuestion - 1)}
                disabled={activeQuestion === 0}
                className={`p-1 rounded ${activeQuestion === 0 ? 'text-gray-300' : 'text-gray-500 hover:bg-gray-100'}`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m15 18-6-6 6-6"/>
                </svg>
              </button>
              <button
                onClick={() => activeQuestion < enhancedQuestions[activeLayer].length - 1 && setActiveQuestion(activeQuestion + 1)}
                disabled={activeQuestion === enhancedQuestions[activeLayer].length - 1}
                className={`p-1 rounded ${activeQuestion === enhancedQuestions[activeLayer].length - 1 ? 'text-gray-300' : 'text-gray-500 hover:bg-gray-100'}`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m9 18 6-6-6-6"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
        
        <div className="p-4 border-b border-gray-200" style={{
          borderLeftWidth: '4px',
          borderLeftStyle: 'solid',
          borderLeftColor: responseLayers[activeLayer].color
        }}>
          <span className="text-xs text-gray-500">Enhanced Question:</span>
          <h3 className="text-gray-800 font-semibold">
            {enhancedQuestions[activeLayer][activeQuestion].enhanced}
          </h3>
        </div>
        
        <div className="p-5 bg-gray-50">
          <span className="text-xs text-gray-500">Recursive Response:</span>
          <p className="text-gray-800 mt-1 leading-relaxed">
            {processContent(filterContentByZoom(enhancedQuestions[activeLayer][activeQuestion].response))}
            {zoomLevel !== 'macro' && zoomLevel !== 'meta' && (
              <span className="text-gray-400 text-sm italic"> [...more]</span>
            )}
          </p>
        </div>
        
        {showInfo && (
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="mb-3">
              <h4 className="font-medium text-gray-700 text-sm mb-2 flex items-center gap-1">
                <span style={{color: responseLayers[activeLayer].color}}>
                  {therapeuticElements[responseLayers[activeLayer].element].symbol}
                </span> 
                Fractal Insights
              </h4>
              <ul className="list-disc pl-5 text-sm text-gray-600">
                {enhancedQuestions[activeLayer][activeQuestion].fractal_insights.map((insight, idx) => (
                  <li key={idx} className="mb-1">{processContent(insight)}</li>
                ))}
              </ul>
            </div>
            
            <div className="mb-3">
              <h4 className="font-medium text-gray-700 text-sm mb-2">Somatic Markers</h4>
              <ul className="list-disc pl-5 text-sm text-gray-600">
                {enhancedQuestions[activeLayer][activeQuestion].somatic_markers.map((marker, idx) => (
                  <li key={idx} className="mb-1">{processContent(marker)}</li>
                ))}
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-700 text-sm mb-2">Bifurcation Points</h4>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="font-medium text-gray-600">Parameter</div>
                <div className="font-medium text-gray-600">Value</div>
                <div className="font-medium text-gray-600">Outcome</div>
                
                {enhancedQuestions[activeLayer][activeQuestion].bifurcation_points.map((point, idx) => (
                  <React.Fragment key={idx}>
                    <div className="text-gray-600">{processContent(point.parameter)}</div>
                    <div className="text-gray-600">{point.value}</div>
                    <div className="text-gray-600">{processContent(point.outcome)}</div>
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Render learning insights if in educational mode */}
      {renderLearningInsights()}
      
      {/* Toggle additional info */}
      <div className="w-full flex justify-center mb-4">
        <button
          onClick={toggleInfo}
          className={`px-3 py-2 rounded-md flex items-center gap-1 ${showInfo ? 'bg-blue-100 text-blue-700 border border-blue-300' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
        >
          <Info size={16} /> {showInfo ? 'Hide Additional Analysis' : 'Show Additional Analysis'}
        </button>
      </div>
      
      {/* Visualization controls */}
      <div className="w-full flex justify-center mb-3">
        <div className="bg-white p-2 rounded-lg border shadow-sm">
          <div className="flex gap-2">
            <button
              onClick={() => handleVisualizationChange('cognitive')}
              className={`px-3 py-2 rounded-md flex items-center gap-1 ${visualizationMode === 'cognitive' ? 'bg-blue-100 text-blue-700 border border-blue-300' : 'text-gray-700 hover:bg-gray-100'}`}
            >
              <Brain size={16} /> Cognitive Architecture
            </button>
            <button
              onClick={() => handleVisualizationChange('fractal')}
              className={`px-3 py-2 rounded-md flex items-center gap-1 ${visualizationMode === 'fractal' ? 'bg-blue-100 text-blue-700 border border-blue-300' : 'text-gray-700 hover:bg-gray-100'}`}
            >
              <Sparkles size={16} /> Fractal Process
            </button>
            <button
              onClick={() => handleVisualizationChange('integration')}
              className={`px-3 py-2 rounded-md flex items-center gap-1 ${visualizationMode === 'integration' ? 'bg-blue-100 text-blue-700 border border-blue-300' : 'text-gray-700 hover:bg-gray-100'}`}
            >
              <Compass size={16} /> Integration Map
            </button>
          </div>
        </div>
      </div>
      
      {/* Visualization canvas */}
      <div className="w-full mb-6">
        <div className="bg-white p-4 rounded-lg border shadow">
          {visualizationMode === 'cognitive' && (
            <canvas 
              ref={cognitiveVisRef}
              width={800} 
              height={400} 
              className="mx-auto border rounded"
            />
          )}
          
          {visualizationMode === 'fractal' && (
            <canvas 
              ref={fractalVisRef}
              width={800} 
              height={400} 
              className="mx-auto border rounded"
            />
          )}
          
          {visualizationMode === 'integration' && (
            <canvas 
              ref={integrationVisRef}
              width={800} 
              height={400} 
              className="mx-auto border rounded"
            />
          )}
        </div>
      </div>
      
      {/* Meta-cognitive evaluation */}
      <div className="w-full bg-white rounded-lg shadow-md p-4 mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
          <RefreshCw size={14} /> Meta-Cognitive System Analysis
        </h3>
        <p className="text-sm text-gray-600 italic">{narrativeGeneration}</p>
        
        {/* NEW: Registry Debug Information */}
        <div className="mt-3 p-2 bg-gray-50 rounded border">
          <h4 className="text-xs font-medium text-gray-600 mb-1">🔧 Registry Debug Info</h4>
          <div className="text-xs text-gray-500">
            <div>Active KnowledgeGraph Instances: {registry.listInstanceIds().join(', ') || 'None'}</div>
            <div>Current Instance ID: mainAriaGraph</div>
            <div>Instance Type: {knowledgeGraphManager.constructor.name}</div>
            <div>Graph Nodes: {knowledgeGraphManager.getGraph().nodes.length}</div>
            <div>Graph Edges: {knowledgeGraphManager.getGraph().edges.length}</div>
          </div>
        </div>
        
        <div className="mt-3 grid grid-cols-3 gap-3">
          {Object.keys(systemMetrics).filter(k => k !== 'lastEvaluationTimestamp').map(metric => (
            <div key={metric} className="bg-gray-50 p-2 rounded">
              <div className="text-xs text-gray-500 mb-1">
                {metric.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </div>
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{width: `${systemMetrics[metric] * 100}%`}}
                  ></div>
                </div>
                <div className="text-xs text-gray-700 w-8 text-right">
                  {(systemMetrics[metric] * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Formula explanation */}
      <div className="w-full bg-white rounded-lg shadow p-4 mb-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Recursive Response Development Model</h3>
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 p-3 bg-gray-50 rounded">
            <p className="font-medium text-gray-700 mb-1">z = Standard Question</p>
            <p className="text-sm text-gray-600">The original survey question as a starting point</p>
          </div>
          <div className="flex-1 p-3 bg-gray-50 rounded">
            <p className="font-medium text-gray-700 mb-1">z² = Recursive Elaboration</p>
            <p className="text-sm text-gray-600">Transformation into multi-layered recursive exploration</p>
          </div>
          <div className="flex-1 p-3 bg-gray-50 rounded">
            <p className="font-medium text-gray-700 mb-1">c = Cognitive Element</p>
            <p className="text-sm text-gray-600">Integration of specific cognitive principles (grounding, integration, etc.)</p>
          </div>
        </div>
      </div>
      
      <div className="text-center text-sm text-gray-500 mt-2">
        <p>This framework transforms standard survey responses into multi-layered recursive explorations</p>
        <p className="text-xs italic mt-1">Mirroring the principles of the Recursive Cognitive Integration Framework</p>
      </div>
    </div>
  );
};

export default RecursiveResponseFramework;