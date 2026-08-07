# The twill command line

This records the palette, the rules, and the degradation ladder for `src/term/`
and `src/cli/`. It exists so that the next person to add a command does not have
to reverse-engineer the taste from the code.

The whole thing is written in twill, in `mode systems`. It does not run yet.
Gaps in the language are recorded in `docs/needs.md`.

## 1. The palette

From the mark: a glowing twisted mint ribbon.

| Name | Hex | Where it is used |
| --- | --- | --- |
| pale | `#D2F0E4` | the ribbon's lit edge; emphasis, quoted source, the shimmer's head |
| mint | `#A8DCCB` | its body; success, the spinner's label at rest |
| accent | `#7FE3C4` | the glow at the twist; headings, the REPL prompt, the spinner glyph |
| teal | `#4FB79B` | its shaded face; rules, brackets, the progress track, agreeing axes |
| deep | `#12332C` | reserved for backgrounds, currently unused |
| ink | `#0B1512` | reserved for backgrounds, currently unused |

Spool, the sibling package manager, is warm. These appear in twill only where
twill is talking about packages, and are borrowed for warnings, so that a
package-shaped concern looks like one across both tools.

| Name | Hex |
| --- | --- |
| spool warm | `#E3A76F` |
| spool pale | `#F2DCC6` |
| spool deep | `#33231A` |

One colour is not in the brand: `#D66A60`, a desaturated red, used only for real
errors. An error has to leave the palette or it stops reading as an error, and a
mint error is a design that has stopped meaning anything. It is desaturated so
that it sits beside mint without shouting.

All of it lives in `src/cli/theme.tw`. No other file names a colour.

## 2. The rules

**Colour carries meaning, never decoration.** Mint is success and structure.
Warm is warning. Red is a real error, at most once per message. If a colour
cannot be given a sentence explaining what it means, it does not go in.

**Colour marks structure, not content.** In the help screen the group titles and
command names are coloured and the descriptions are not. Colouring the
descriptions makes a page where every line is loud and the shape of the page is
invisible.

**Four levels of emphasis, no more.** heading, body, muted, emphasis. Body is
deliberately unstyled so it inherits the user's own foreground and therefore
works on a light theme, which no hardcoded colour can do.

**Animation is for things that take time.** The spinner does not appear until
90ms have passed, so a fast path through a command produces no animation and no
cursor games. An indicator that appears and vanishes inside one frame is worse
than none: the eye catches the flash and has nothing to attach it to.

**Nothing blinks.** No SGR 5 anywhere, and no effect that takes a character
from visible to invisible. The spinner turns; the shimmer moves hue along a
label without any character going dark. Liveness comes from motion, not from
flashing.

**Slow is calmer than fast.** The spinner is 120ms a frame, a half turn a
second. Repaints are capped at 30 a second. Faster is invisible to the eye and
costs bytes on every link between here and it, which is what makes a remote
session feel sluggish.

**Meaning never lives only in colour.** Every status has a glyph as well as a
colour. The failing axis in a shape error has a caret under it as well as being
red. Turn colour off and nothing is lost but the speed of finding it.

**Never repaint identical bytes.** `frame.tw` compares against the last body and
sends nothing when they match, because repainting the same content is pure
flicker on terminals that do not double-buffer.

**Truncate every line in a frame.** A line that wraps breaks the repaint
arithmetic permanently: the frame counts logical lines and the terminal counts
visual ones. This is not a nicety.

**The banner appears twice.** Bare `twill`, and `twill --help`. Nowhere else.

**Exit codes are the API.** Anything that renders an error exits non-zero. A
rendered diagnostic with exit 0 is a broken build that CI calls green.

## 3. The degradation ladder

Capability is detected once in `src/term/caps.tw` and every other file obeys it.
The bias is one-directional: when a signal is ambiguous, assume less. A plain
run in a capable terminal is a missed flourish. An escape sequence in a log file
is a bug that survives for years, because the person who caused it never sees
it.

### Colour tiers

| Tier | Condition | What changes |
| --- | --- | --- |
| truecolor | `COLORTERM=truecolor`/`24bit`, or a known terminal | full palette, gradients, the shimmer |
| 256 | `TERM` contains `256color`, or a tty that claimed nothing | palette quantised per channel to the 6x6x6 cube plus the 24-step grey ramp; the shimmer is dropped and the label is flat mint, because a quantised gradient reads as a defect |
| plain | `NO_COLOR` set, `TERM=dumb`, `TERM` empty, `--no-color`, or stdout is not a tty | every styling function returns the empty string; the output is exactly the text it would have been |

`NO_COLOR` is honoured on presence, whatever the value. `FORCE_COLOR` is the
only way to raise the tier, and it exists for logs that are replayed through a
renderer.

### Everything else

| Signal | On | Off |
| --- | --- | --- |
| tty | cursor control, in-place repainting, prompts | no cursor sequences at all; spinner emits nothing and prints one outcome line; progress bar prints one line per ten percent; REPL prints no prompt |
| unicode | braille mark, block-fill progress, `✓ ▲ ✕ ❯ ━ ⋮ …` | `ok ! x > ^ : ...`, `#` and `.` for the bar |
| OSC 8 hyperlinks | `file:line:col` in a diagnostic is clickable | the same text, unlinked; the label already contains the path so nothing is lost |
| width | layouts fill the terminal, tensors show as many columns as fit | 80 columns assumed; below the minimum the progress bar drops rather than squeezing |

### Widget by widget

- **banner** needs braille *and* colour *and* a tty *and* 40 columns. Missing
  any of them it degrades to one line: `twill 0.28.0`. There is no middle
  rendering, because a monochrome braille blob is not a logo.
- **spinner** degrades: truecolor shimmer, then flat mint with a turning glyph,
  then an ASCII glyph, then (no tty) silence plus one line when the work ends.
- **progress bar** degrades: gradient eighth-blocks, then flat `#`, then one
  milestone line per ten percent. Not one line per step, which buries a build
  log, and not nothing, because a CI job silent for six minutes is
  indistinguishable from a hung one.
- **diagnostics** degrade only in colour. The gutter, the caret span and the
  layout are drawn from ASCII-safe characters where unicode is missing, and the
  information content is identical at every tier.

## 4. The shape mismatch

This is the error the tool is judged on, because shape errors are what twill is
for. `src/cli/shape.tw`.

Both shapes are laid out as a table, one axis per column, **aligned from the
right**. Right, because that is how broadcasting aligns them; a rendering that
aligned them left would be drawing a different relation from the one being
checked. Column widths are computed across both shapes so the rows line up
character for character, which is the entire reason it works.

Then:

- agreeing axes in quiet teal;
- an axis of length 1 that will stretch in warm, because that is a thing worth
  noticing and not an error;
- a missing leading axis as `·` in muted, distinct from a length-1 axis, because
  they are different mistakes and merging them makes the user fix the wrong one;
- the **first** disagreeing axis in bold red, with a caret under it and its
  index spelled out. First, not all, because one transposition usually produces
  several and reporting five reads as five problems.

The help line names the change to make. It detects the transposition case, where
the two disagreeing dimensions each appear on the other side one axis over, and
says so, which turns a five-minute hunt into a one-line fix.

## 5. Layout of the code

```
src/term/     primitives, no twill-specific knowledge
  caps.tw     capability detection, the one place that decides
  color.tw    Rgb, the 256 approximation, interpolation and ramps
  ansi.tw     escapes, Style, cursor control, OSC 8
  width.tw    escape-aware measuring, truncation, wrapping, tab expansion
  frame.tw    in-place repainting

src/cli/      the experience
  theme.tw    the palette and the role each colour plays
  banner.tw   the ribbon mark, computed from the twist
  spinner.tw  indeterminate work
  progress.tw determinate work
  diagnostic.tw  file:line:col, source, caret, explanation
  shape.tw    the shape mismatch
  help.tw     grouped commands, one aligned description column
  tensor.tw   aligned columns, elided middles
  repl.tw     the prompt and the session
  main.tw     dispatch, and the three policies
```

Nothing in `src/term/` writes to stdout. Every function returns a string. That
is what lets the whole UI be rendered into a buffer and diffed in a test, and it
is why the plain path costs nothing: at the plain tier the styling functions
return `""` and the caller's concatenation collapses to the text it would have
printed anyway.

## 6. Things deliberately not done

- **No alternate screen.** A tool that clears your scrollback to show a progress
  bar and hands it back afterwards is a tool whose output you cannot read later.
- **No `out[3] =` result labels in the REPL.** Nobody refers back by index in a
  terminal, and the label costs a column of every line forever.
- **No box drawing around anything.** Borders cost two columns and four lines
  and add no information. Alignment and whitespace do the same job.
- **No spinner on fast commands, and no banner on working commands.** These are
  the two rules most likely to be broken by the next person in a hurry, which is
  why `main.tw` makes both an explicit decision rather than a default.
