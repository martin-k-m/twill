<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/martin-k-m/twill/main/assets/twill-mark-glow.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/martin-k-m/twill/main/assets/twill-mark.png">
    <img alt="twill" src="https://raw.githubusercontent.com/martin-k-m/twill/main/assets/twill-mark.png" width="72">
  </picture>
</p>

<h1 align="center">twill documentation</h1>

Everything written down about the language, grouped by what you came here to do.
The [README](../README.md) is the tour; this is the index.

## Learning the language

| Document | What it is |
| --- | --- |
| [tutorial.md](tutorial.md) | From nothing to a trained model. Start here if you have never run a `.tw` file. |
| [language-guide.md](language-guide.md) | The reference. Syntax, builtins, shapes, units, the standard library. Short, because the language is small. |

## Understanding the design

| Document | What it is |
| --- | --- |
| [design.md](design.md) | Why twill is built the way it is, the principles it holds to, and the roadmap. |
| [finance.md](finance.md) | Where twill can beat a Python stack for financial ML, and where it cannot. Written as an assessment, not a pitch. |
| [gpu-feasibility.md](gpu-feasibility.md) | Should twill get a GPU backend? Measured on one machine, with the numbers. The answer is not yet, and the reasoning is the useful part. |

## The implementation, and where it is going

The reference implementation is Go, under `internal/`. The next one is twill,
under `src/`, and it does not run yet. These four documents are that effort.

| Document | What it is |
| --- | --- |
| [self-hosting.md](self-hosting.md) | The design. `mode systems`, the file-level mode mechanism, and what each systems feature costs the numeric language. |
| [type-system.md](type-system.md) | The stage 2 design. Function types, generics unified with the existing shape variables, `enum` with exhaustive `match`, `Opt`/`Res` with `?`, and generic containers, as one system. Says what it refuses to add and why. |
| [needs.md](needs.md) | The work queue. One numbered entry per feature `src/` reaches for and the bootstrap does not provide, naming the file and line. Ordered by dependency. |
| [roadmap.md](roadmap.md) | The argued ranking. Every missing feature ordered by how many of the six twill codebases hit it independently, with the workaround each one uses and what it costs, grouped into stages. |
| [rewrite-plan.md](rewrite-plan.md) | The earlier plan to rewrite out of Go into Rust. Superseded by `self-hosting.md` on the target question only; its staging discipline, differential harness and fixture corpus all still apply. |
| [cli-design.md](cli-design.md) | The palette, the rules and the degradation ladder for the twill command line, so the next command added does not have to reverse-engineer the taste from the code. |

## Assets

| Document | What it is |
| --- | --- |
| [brand.md](brand.md) | The mark, the palette, which asset goes where, clear space and minimum size. Read before touching anything in `assets/`. |

## Reading order

If you want the whole picture and have an afternoon:

1. [tutorial.md](tutorial.md), to get a program running.
2. [language-guide.md](language-guide.md), skimmed, for what exists.
3. [design.md](design.md), for why.
4. [self-hosting.md](self-hosting.md) then [needs.md](needs.md), for where it goes
   next. Those two are the most active part of the project.
5. [roadmap.md](roadmap.md), for what gets built first and why.
6. [type-system.md](type-system.md), for the design of the stage the roadmap
   ranks first.
