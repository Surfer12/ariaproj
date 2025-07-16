# System Architecture Diagram

## AI Problem-Solving System - Component Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              User Input / Problem Statement                          │
└─────────────────────────────────┬───────────────┬───────────────────────────────────┘
                                  │               │
                    ┌─────────────▼─────┐   ┌─────▼─────────────┐
                    │  Symbolic Module  │   │   Neural Module   │
                    │                   │◄──┤                   │
                    │ • Formal Logic    │   │ • Pattern Match   │
                    │ • Theorem Proving │   │ • Heuristics      │
                    │ • SAT/SMT Solvers │   │ • GNN/Transformer │
                    └─────────┬─────────┘   └─────────┬─────────┘
                              │ S(x)                  │ N(x)
                              │                       │
                              └───────────┬───────────┘
                                         │
                            ┌────────────▼────────────┐
                            │ Hybrid Blending         │
                            │   Mechanism             │
                            │                         │
                            │ Result = α·S(x) +      │
                            │         (1-α)·N(x)     │
                            └────────────┬────────────┘
                                        │ Blended Result
                                        │
                            ┌───────────▼────────────┐
                            │ Bayesian               │
                            │ Regularization Engine  │
                            │                        │
                            │ • Cognitive (λ₁)       │
                            │ • Efficiency (λ₂)      │
                            └───────────┬────────────┘
                                       │ Regularized
                                       │
                            ┌──────────▼─────────────┐
                            │ Cognitive Bias         │
                            │    Modeler             │
                            │                        │
                            │ • Human Biases (β)     │
                            │ • Interpretability     │
                            └──────────┬─────────────┘
                                      │ Biased & Interpretable
                                      │
                            ┌─────────▼──────────────┐
                            │ Explanation            │
                            │   Generator            │
                            │                        │
                            │ • Natural Language     │
                            │ • Step Attribution     │
                            └─────────┬──────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Interactive Control Interface                                │
│                                                                                     │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     ┌──────────────────────────┐ │
│  │   α    │  │   β    │  │   λ₁   │  │   λ₂   │     │   Reasoning Trace       │ │
│  │ ──●─── │  │ ──●─── │  │ ──●─── │  │ ──●─── │     │   Visualization         │ │
│  │ 0   1  │  │ 0   1  │  │ 0  10  │  │ 0  10  │     │                         │ │
│  └────────┘  └────────┘  └────────┘  └────────┘     └──────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            Validation Benchmark                                      │
│                                                                                     │
│  • Accuracy Testing        • User Feedback Collection                              │
│  • Robustness Checks      • Performance Metrics                                    │
│  • Safety Validation      • A/B Testing                                            │
└─────────────────────────────────────┬───────────────────────────────────────────────┘
                                     │ Feedback
                                     │
                            ┌────────▼───────────────┐
                            │ Meta-Optimization      │
                            │    Controller          │
                            │                        │
                            │ • Auto-tune Parameters │
                            │ • Bayesian Optimization│
                            │ • User Overrides      │
                            └────────────────────────┘
                                     │
                                     │ Parameter Updates
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
          [Update α]          [Update λ₁,λ₂]       [Update β]
```

## Key Data Flows

### Primary Processing Path:
1. **Input** → Symbolic & Neural Modules (parallel processing)
2. **Module Outputs** → Hybrid Blending
3. **Blended Result** → Regularization
4. **Regularized Output** → Bias Modeling
5. **Biased Output** → Explanation Generation
6. **Explanation** → User Interface

### Control Flows:
- **User Controls** → Direct parameter adjustment (α, β, λ₁, λ₂)
- **Validation Results** → Meta-Optimization Controller
- **Meta-Optimizer** → Automatic parameter tuning

### Feedback Loops:
- **Neural ↔ Symbolic**: Heuristic guidance exchange
- **User Feedback** → Validation → Meta-Optimization
- **Parameter Updates** → All processing modules

## Module Interactions

### Cooperative Relationships:
- **Symbolic + Neural**: Complementary reasoning approaches
- **Blending + Regularization**: Optimization pipeline
- **Bias Modeling + Explanation**: Human-centric output

### Control Relationships:
- **Meta-Optimizer → Processing Modules**: Parameter control
- **User Interface → All Modules**: Manual override capability
- **Validation → Meta-Optimizer**: Performance-based tuning

### Information Flow Types:
- **Solid Lines (─)**: Primary data flow
- **Arrows (→)**: Direction of information
- **Bidirectional (↔)**: Two-way communication