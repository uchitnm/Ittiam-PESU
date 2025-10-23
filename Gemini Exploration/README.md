# Soccer Highlight Generator

A two-level prompt design framework for progressive video analysis, demonstrating how contextual understanding transforms raw visual perception into semantic event interpretation.

## Overview

This project explores a **hierarchical prompt architecture** that progresses from domain-agnostic visual perception (L0) to soccer-specific highlight extraction (L1). The framework tests how large language models interpret and structure visual information when gradually introduced to **domain context** and **task-specific objectives**.

The two-level approach provides insights into:
- How context shapes interpretation of the same visual input
- The transition from "what is visible" to "what it means"
- Scalable methods for automated sports highlight generation

---

## Architecture

### L0 — General Visual Understanding

**Objective:** Establish a baseline of pure visual perception without domain knowledge.

**Design Principles:**
- Model receives a **silent video clip** with no contextual information
- Must describe events **objectively** using only motion, interaction, and sequence
- No domain assumptions permitted (e.g., cannot assume it's a soccer match)

**Purpose:**
- Tests **raw perception capabilities** — action recognition, transition detection, visual cue identification
- Ensures descriptions remain **neutral**, **scene-based**, and **context-free**
- Provides a baseline for measuring the impact of contextual grounding

**Example Output:**

```
00:00 – 00:03 | A group of people run across a field
00:03 – 00:07 | One person passes a round object to another
00:07 – 00:10 | Another person moves quickly toward an opening
```

---

### L1 — Soccer Highlight Generator

**Objective:** Apply domain-specific understanding for timestamp-wise event analysis and highlight extraction.

**Design Principles:**
- Model now understands the clip shows a **soccer match**
- Analyzes video **frame-by-frame** to identify:
  - **Events** (pass, tackle, shot, save, foul, goal, etc.)
  - **Outcomes** (successful, failed, saved, scored)
  - **Highlight Classification** (`Routine`, `Notable`, `Highlight`)
- Output format: structured, chronological, concise

**Purpose:**
- Tests **contextual comprehension** and **discriminative reasoning**
- Shifts from observation to **tactical and event-based analysis**
- Enables automated highlight generation workflows

**Example Output:**

```
00:00 – 00:04 | Midfielder intercepts and passes forward | Successful pass | Notable
00:04 – 00:08 | Striker dribbles past defender and shoots | Saved by goalkeeper | Highlight
00:08 – 00:10 | Corner cleared by defense | Possession regained | Routine
```

---

## Design Philosophy

| Level | Context Provided | Understanding Type | Output Style | Purpose |
|-------|------------------|-------------------|--------------|---------|
| **L0** | None (Domain-Agnostic) | Visual perception & sequencing | Neutral narration | Baseline visual comprehension |
| **L1** | Soccer match context | Domain reasoning & event detection | Timestamped structured log | Contextual highlight extraction |

---

## Key Insights

This dual-level framework demonstrates how **contextual grounding fundamentally transforms video interpretation**:

- **L0 → Perception Layer**: Focuses on *what is visible* (motion, objects, sequences)
- **L1 → Semantic Layer**: Focuses on *what it means* (events, outcomes, significance)

The progression illustrates:
1. How the same visual input yields dramatically different interpretations based on context
2. The relationship between domain knowledge and semantic understanding

---

## Use Cases

- **Sports Highlight Generation**: Automated extraction of key moments from match footage
- **Video Summarization**: Context-aware content condensation for efficient review
- **Automated Editing**: AI-driven identification of clip-worthy segments
- **Performance Analysis**: Tactical event logging for coaching and review
- **Model Evaluation**: Benchmarking multimodal understanding in vision-language models

