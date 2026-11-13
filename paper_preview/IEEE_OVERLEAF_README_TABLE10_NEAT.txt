IEEE OVERLEAF PROJECT
=====================

1. In Overleaf select New Project > Upload Project.
2. Upload battery_pinn_ieee_table10_neat_overleaf.zip.
3. Open Menu and set Main document to battery_pinn_ieee_conference.tex.
4. Set Compiler to pdfLaTeX.
5. Select the newest TeX Live version and choose Recompile from scratch.

The ZIP contains:

  battery_pinn_ieee_conference.tex
  battery_pinn_ieee_conference_strict.tex
  figures/pinn_architecture.png
  figures/experimental_design.png
  figures/inference_efficiency_protocol.png

The manuscript begins with the supplied IEEE conference template structure:

  \documentclass[conference]{IEEEtran}
  \IEEEoverridecommandlockouts
  \usepackage{cite}
  \usepackage{amsmath,amssymb,amsfonts}
  \usepackage{algorithmic}
  \usepackage{graphicx}
  \usepackage{textcomp}
  \usepackage{xcolor}

The five authors share one affiliation block because all authors belong to the
same department and institution. This follows the template's instruction to
keep identical affiliations succinct and keeps the complete paper within the
eight-page conference limit.

Do not add geometry, fullpage, setspace, times, or manual font-size commands.
The IEEEtran class controls the official margins, columns, body type, headings,
captions, and references.

HEADING VERSION
---------------

The main file uses a clearer three-level heading hierarchy:

  Main section headings: 12 pt bold, centered
  Subsection headings:    10 pt bold
  Subsubsection headings: 10 pt bold italic

The body remains 10 pt, the abstract and index terms are 9 pt, author names are
approximately 11 pt, and the title is approximately 24 pt. IEEEtran selects a
Times-compatible Roman family when compiled with pdfLaTeX on Overleaf.

The 12 pt section size follows the stronger visual hierarchy requested from the
sample paper while preserving the 10 pt IEEE body text. If a conference requires
the untouched IEEEtran style, set the main document to
battery_pinn_ieee_conference_strict.tex instead.

AUTHOR ORDER
------------

1. Ms. Sasikala C
2. Naresh V
3. Mohamed Shaheel A
4. Kishore J
5. Kishore P. S.

Table I compares exactly ten published studies from 2023--2026. It covers SPM
and P2D surrogates, electrochemical parameter inference, temperature and heat-
map prediction, SOC estimation, digital twins, and multiphysics aging models.
The proposed model is discussed separately and is not counted as one of the ten
papers. The bibliography contains 26 cited sources; 14 were published in
2024--2026 (seven from 2024, five from 2025, and two from 2026). Older sources
are retained only where they establish the governing equations or software.
The References
heading begins in the left column on page 8 and entries continue into the right
column through normal IEEE two-column flow. The final columns are balanced for
a cleaner last page. A small, visible inter-entry gap improves readability while
keeping the manuscript at exactly eight pages. Do not place the bibliography in
a one-column minipage or add uncited sources merely to remove normal white space.

TABLE I REVISION
----------------

The abstract and related-work section now state the literature-based gap:
multiphysics coupling alone is not new. The paper tests the narrower hypothesis
that deriving particle stress from the predicted concentration improves cross-
field consistency under sparse labels. Table I records each study's main task,
physics/model, and the specific difference from this work.

The final Table I restores the six-column comparison used in the earlier
five-paper version:

  Study | Main task | Physics represented | Thermal state |
  Mechanics/aging | Key distinction

Each entry is reduced to short evidence-based phrases. Bold text highlights
the thermal/mechanical coverage and the principal difference from the proposed
model. Small inter-row spacing prevents the ten-study table from appearing as
one dense paragraph while keeping the full manuscript at eight pages.

EVALUATION ADDITION
-------------------

The paper now specifies nine primary metrics. The added metric is median PINN
inference speedup over the PyBaMM SPMe solver. Figure 3 documents the timing
protocol: identical held-out query batches and hardware, 100 warm-up calls,
synchronization, repeated timings, and reporting of median latency, P95 latency,
and speedup. The result table contains a pending speedup column. Replace the
pending values only after timing both methods under this protocol.

The numerical result macros remain marked "pending" until the complete
three-seed experiment is run. Replace them only with values exported by the
accompanying notebook.
