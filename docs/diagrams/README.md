# Diagrams

Mermaid sources for every diagram in `docs/`, plus PNG exports of the two
that slides need most. GitHub renders the `.mmd` content inline where it
appears in the documents; the files here are the same blocks, extracted so
they can be pasted into https://mermaid.live or a slide tool.

| file | appears in | what it shows |
|---|---|---|
| `system.mmd` / `system.png` | [00_OVERVIEW.md](../00_OVERVIEW.md) | sense → classify → mesh/escalate → Foresight → human-approved alert, built vs planned |
| `architecture_modules.mmd` | [01_ARCHITECTURE.md](../01_ARCHITECTURE.md) | execution order of every package, mapped to code |
| `escalation_ladder.mmd` | [01_ARCHITECTURE.md](../01_ARCHITECTURE.md) | Tier 0 → 1 → 2 state machine with the implemented thresholds |
| `ignition_sequence.mmd` | [01_ARCHITECTURE.md](../01_ARCHITECTURE.md) | one ignition from first sensor sample to CAP draft, with the canonical demo's timestamps |
| `data_flow.mmd` / `data_flow.png` | [02_DATA_PIPELINE.md](../02_DATA_PIPELINE.md) | every dataset and the synthetic/real boundary |
| `roadmap.mmd` | [05_ROADMAP.md](../05_ROADMAP.md) | now / next / later |

Regenerate the PNGs (matplotlib only; the repository has no Mermaid renderer):

```bash
python docs/diagrams/render_diagrams.py
```

`render_diagrams.py` mirrors the content of `system.mmd` and `data_flow.mmd`
by hand. If you change a `.mmd`, change the matching inline block in the
document and, for those two, the renderer as well.

Styling convention in every diagram: solid green = built and tested in this
repository as simulation; dashed amber = PLANNED or NOT BUILT.
