# Compressed Agent

An agent system that stores all reasoning in a compressed externalized latent state using typed graphs, factors, and canonical micro-summaries.

## Architecture

### Core Components

1. **Latent State** (`latent_state.py`)
   - Typed graph with nodes and edges
   - Factors representing constraints/relationships
   - Canonical micro-summaries for compression
   - All state is versioned and timestamped

2. **State Edit Pipeline** (`state_edit.py`)
   - **Normalize**: Convert state to canonical form (deduplicate, canonicalize IDs)
   - **Propose**: Generate proposed edits based on goals
   - **Apply**: Apply edits to update state
   - Each step is a compressed state-edit instead of full natural-language replanning

3. **Tool Interface** (`tools.py`)
   - Tools read/write the minimal latent substrate
   - Tools return state edits rather than full state descriptions
   - Includes: QueryTool, CreateNodeTool, CreateEdgeTool, CreateSummaryTool

4. **Agent Orchestrator** (`agent.py`)
   - Multi-turn planning using compressed state
   - Stable planning across multiple turns
   - Radically reduced token usage through compression

## Usage
### Single file test
```bash
uv run python src/benchmarks/run_benchmark_with_score.py
```
### All files test
```bash
uv run python src/benchmarks/aggregate_scores.py
```