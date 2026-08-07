# Brand

The twill mark is a ribbon: it sweeps right along the top, folds back on itself
in the middle so the underside shows as a darker face, and tapers to a point at
the bottom left. Two features carry the identity and must survive any edit or
re-export: the fold, and the concave notch on the right under the top sweep.
A filled silhouette without them is not the mark.

## Palette

| Role   | Hex       | Used for                                             |
| ------ | --------- | ---------------------------------------------------- |
| pale   | `#D2F0E4` | top of the sweep; wordmark text on dark backgrounds  |
| mint   | `#A8DCCB` | middle of the gradient                               |
| accent | `#7FE3C4` | bottom of the sweep; the bloom on the glow variant   |
| teal   | `#4FB79B` | the folded-under faces                               |
| deep   | `#12332C` | dark backgrounds, VS Code gallery banner             |
| ink    | `#0B1512` | body text on light backgrounds                       |

Spool, the package manager, is the warm sibling and does not use these:
`#E3A76F` (accent), `#F2DCC6` (pale), `#33231A` (deep).

## Which asset where

| File                        | Use                                                       |
| --------------------------- | --------------------------------------------------------- |
| `twill-mark.svg`            | the default. Anything that can render SVG.                |
| `twill-mark-flat.svg`       | badges, buttons, and anywhere the mark should take the surrounding text colour. Fills with `currentColor`. |
| `twill-mark-glow.svg`       | dark backgrounds only, when the bloom is wanted.          |
| `twill-wordmark.svg`        | mark plus word, on light backgrounds.                     |
| `twill-wordmark-dark.svg`   | mark plus word, on dark backgrounds.                      |
| `twill-mark.png` and sizes  | raster contexts: READMEs, the VS Code extension icon, anywhere SVG is not accepted. |
| `twill-mark-glow.png`       | the dark `<picture>` source in READMEs.                   |
| `favicon.ico`               | site favicon, multi-size.                                 |
| `twill-mark-180.png`        | `apple-touch-icon`.                                       |
| `twill-source.png`          | the original render. Reference only, never shipped.       |

`currentColor` in `twill-mark-flat.svg` only resolves when the SVG is inlined in
the page or used as a CSS mask. Referenced through `<img>` or `<object>` it
falls back to black, because the SVG is then a separate document with no colour
to inherit.

## Light and dark

On light backgrounds use `twill-mark.svg` or `twill-mark.png` as they are; the
gradient has enough weight to hold against white.

On dark backgrounds prefer `twill-mark-glow.*`. The plain mark also works on
`#12332C`, but the bloom is what the original render looks like and it is worth
using where there is room for it. Never put the glow variant on a light
background: the bloom turns to grey haze around the edges.

Never recolour the mark outside the palette above, never add a drop shadow on
top of the glow, and never place the mark on a photograph or a busy background
without a solid plate behind it.

## Clear space and minimum size

Clear space on all four sides is one quarter of the mark's height. Nothing else
goes inside that: no text, no border, no other logo.

For the wordmark, the gap between the mark and the "t" is part of the lockup and
must not be changed. Set the whole file, do not rebuild it from parts.

Minimum size is 16px for the vector and 48px for the raster.

**Below 48px, use the vector.** The raster mark dissolves under about 48px: the
fold and the notch smear into a single blob and what is left is unrecognisable.
The `-32`, `-16` PNGs exist only for contexts that cannot take an SVG at all,
such as `favicon.ico`.
