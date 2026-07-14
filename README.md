# local-multi-agent-code-generator

Local multi-agent code generator. Uses a **Prompt Engineer, Planner, Generator, Reviewer & Syntax Checker** working together to turn vague prompts into working Python scripts. Self-healing, crash-proof, and built to run reliably on modest hardware with [Ollama](https://ollama.com). No API keys needed.

## Sample Command to Execute a Task

You can command the agent to execute a coding task using the `--prompt` argument:

```bash
python main.py --prompt "Write a Python script to fetch the current temperature of any city using a public API."
```

Or run it interactively (it will ask for your prompt):

```bash
python main.py
```

Other useful flags:

| Flag | Short | Description |
|---|---|---|
| `--model` | `-m` | Ollama model to use (default: `qwen2.5-coder:1.5b`) |
| `--output` | `-o` | Custom output path for the generated `.py` file |
| `--fast` | `-f` | Skip planning stage for quick single-shot generation |
| `--install` | `-i` | Auto-install missing packages without prompting |

### ⚡ Fast Mode (`--fast` / `-f`)

By default the agent runs a **full 4-step pipeline** (Interpreter → Planner → Generator → Reviewer). Fast mode skips the Interpreter and Planner and goes straight to code generation in a single shot — much quicker, but with less structured output.

```bash
# Standard mode (recommended — best quality output)
python main.py --prompt "Write a binary search algorithm in Python."

# Fast mode (quick and lightweight — skips planning)
python main.py --fast --prompt "Write a binary search algorithm in Python."
```

| | Standard Mode | Fast Mode (`--fast`) |
|---|---|---|
| **Steps** | Interpreter → Planner → Generator → Reviewer → Checker | Generator → Checker |
| **Quality** | Higher — structured, reviewed code | Variable — single-shot output |
| **Speed** | Slower (more LLM calls) | Faster (fewer LLM calls) |
| **Best for** | Complex tasks, algorithms, multi-step logic | Quick scripts, simple tasks |

> [!TIP]
> Use **fast mode** for simple one-off scripts where speed matters. Use the **standard pipeline** for anything with algorithm logic, multiple components, or specific library requirements.

---


## How Agents Talk to Each Other

The pipeline isn't a silent black-box — every agent announces its work in a styled, colour-coded dialogue box in your terminal, so you can follow the collaboration in real time:

```
╔══════════════════════════════════════════════════════════════╗
║──────────────── 🧠 Supervisor ───────────────────────────────║
╠══════════════════════════════════════════════════════════════╣
║  Pipeline starting. Model: qwen2.5-coder:1.5b | Fast: False ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║──────────────── ✏️  Interpreter ─────────────────────────────║
╠══════════════════════════════════════════════════════════════╣
║  Analysing your request. Expanding it into a detailed,       ║
║  unambiguous technical prompt for the Planner...             ║
╚══════════════════════════════════════════════════════════════╝
```

The Reviewer and Generator engage in a **multi-turn dialogue loop** (up to 3 rounds). If the Reviewer finds issues, the Generator reads the *full accumulated conversation* before revising — not just the last error — so corrections get progressively smarter:

```
Reviewer  →  "Issues found: wrong algorithm used."
Generator →  "Acknowledged. Revising code to address all raised issues..."
Reviewer  →  "Code is fully aligned. Approving."
```

### Agent Roles

| Agent | Emoji | Responsibility |
|---|---|---|
| **Supervisor** | 🧠 | Orchestrates the full pipeline, manages retries and output |
| **Interpreter** | ✏️ | Expands your vague prompt into a precise technical specification |
| **Planner** | 📋 | Designs the step-by-step implementation plan |
| **Generator** | ⚙️ | Writes the actual Python code from the plan |
| **Reviewer** | 🔍 | Checks that the code actually matches what you asked for |
| **Checker** | ✅ | Validates syntax and detects missing package dependencies |

---

> [!NOTE]
> **Designed for reliability on limited hardware (8GB RAM)**
>
> This project is intentionally optimised for low-resource, local-first use. It defaults to a small, fast model (`qwen2.5-coder:1.5b`) that fits comfortably in 8GB RAM via Ollama. You're trading raw model capability for zero cloud dependency, zero API costs, and full privacy.
>
> **Honest notes for users:**
>
> * **Single-file output**: Each run generates one self-contained `.py` file. Multi-file project scaffolding is not yet supported.
> * **Static validation only**: Code is checked for syntax and imports — it is not executed in a sandbox. Runtime bugs can still slip through.
> * **Small model limitations**: Tiny LLMs can misinterpret complex constraints. The multi-turn review loop is specifically designed to catch and correct these cases.
> * **Community-stage project**: Solo-authored, early-stage. Works well for personal scripting tasks. Not yet battle-tested for production use.
>
> **Bottom line**: It works well for personal automation, learning, and quick prototyping on your own machine. Treat generated code as a solid first draft, review before running anything sensitive.

