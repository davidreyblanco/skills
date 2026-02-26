# Agent Skills Playground

Examples and experiments for building reusable `SKILL.md` packages aligned with the official Agent Skills standard.

- Integration guide: https://agentskills.io/integrate-skills
- Specification: https://agentskills.io/specification

## Purpose

This repository is a small test bed to:

- Prototype and iterate on Agent Skills.
- Keep skills in a portable format that can be reused across compatible agents.
- Validate structure and metadata against the public `agentskills.io` spec.

## Repository Structure

```text
.
├── data-science-trainer/
│   ├── SKILL.md
│   └── LICENSE.txt
├── LICENSE
└── README.md
```

Each skill lives in its own folder and must include a `SKILL.md` file.

## Compliance Notes (agentskills.io)

The current setup follows the core format expectations:

- One directory per skill.
- Required `SKILL.md` entry point.
- YAML frontmatter with required fields:
  - `name`
  - `description`
- Markdown body with executable instructions for the agent.
- Optional metadata fields (for example: `license`, `compatibility`, `metadata`) when useful.

Recommended for continued compliance:

- Keep `name` concise (max 64 chars).
- Keep `description` clear and activation-focused.
- Keep `SKILL.md` readable and scoped; move heavy detail into referenced files when needed.

## How To Add a New Skill

1. Create a new folder (example: `my-new-skill/`).
2. Add `my-new-skill/SKILL.md`.
3. Include YAML frontmatter and instructions, for example:

```md
---
name: my-new-skill
description: Short description of what this skill does and when to use it.
---

# My New Skill

## When to use
Use this skill when...

## Steps
1. ...
2. ...
```

4. Add optional folders only if needed:
   - `scripts/`
   - `references/`
   - `assets/`

## Validation

If you use the reference tooling:

```bash
skills-ref validate ./data-science-trainer
```

You can also generate prompt metadata blocks (for skill discovery in agents):

```bash
skills-ref to-prompt ./data-science-trainer
```

## Current Skill

- `data-science-trainer`: guidance for designing and developing technical data science/ML training programs.

## License

Repository-level license: see `LICENSE`.
Skill-specific license/additional terms: see each skill folder (for example, `data-science-trainer/LICENSE.txt`).
