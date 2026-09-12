# Project Roadmap: DropIn
**Evaluating Multi-Agent Systems for Proactive Academic Advising**

## Overview
A 10-week technical execution plan to develop and evaluate an AI-native proactive advising engine using multi-agent orchestration, the Model Context Protocol (MCP), and the Gemini API.

## 10-Week Execution Timeline

| Week | Phase | Key Tasks & Milestones | Deliverable |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Setup & Infrastructure** | Initialize local repo; install LangGraph, Gemini GenAI SDK, and MCP libraries; implement rate-limiting backoff. | Local development environment configured. |
| **Week 2** | **Synthetic Data Engineering** | Define database schema; generate 100 realistic student profiles with varying risk levels via Gemini. | SQLite/CSV mock database populated. |
| **Week 3** | **Baseline Implementation** | Engineer a unified single-LLM prompt architecture to process student data and output intervention plans. | Functional single-LLM evaluation script. |
| **Week 4** | **Multi-Agent: Data Retrieval** | Develop the Data Retrieval Agent; implement MCP to autonomously query the synthetic database. | Agent capable of hallucination-free data fetching. |
| **Week 5** | **Multi-Agent: Analysis & Planning** | Develop the Academic Analysis Agent and Intervention Planning Agent. | Specialized agents for reasoning and drafting plans. |
| **Week 6** | **Agent Orchestration** | Link agents into a LangGraph pipeline; implement state management for secure data handoffs. | Functional multi-agent pipeline prototype. |
| **Week 7** | **Standardized Testing** | Process 20 synthetic scenarios through both the baseline and DropIn multi-agent pipelines. | Raw output logs and execution traces. |
| **Week 8** | **Benchmarking & Evaluation** | Analyze decision accuracy, tool usage success (MCP), token efficiency, and pipeline latency. | Data analysis and comparative visualization charts. |
| **Week 9** | **Research Report Drafting** | Draft the formal evaluation paper covering methodology, system architecture, and results. | First draft of the final research report. |
| **Week 10** | **Refinement & Submission** | Debug edge cases; finalize the architectural trade-off documentation; prepare final deliverables for Canvas. | **Completed codebase and final report.** |

## Core Technology Stack
* **LLM Engine:** Google AI Studio (Gemini 2.5 Flash / Gemini Pro)
* **Orchestration:** LangGraph
* **Tool Integration:** Model Context Protocol (MCP)
* **Environment:** VS Code, Python 3.x
* **Database:** Local SQLite / CSV (Synthetic Data)