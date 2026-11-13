# Local overlap audit of the revised PINN battery paper

## How much exact text from the earlier manuscript remains?

### Takeaway
The revised PDF is substantially different in its abstract, introduction, related-work narrative, results discussion, limitations, and conclusion, but it is not a low-overlap document when compared with the earlier PDF as a whole. Using exact normalized token sequences, 43.34% of the revised document is covered by matches of 12 or more consecutive words; after removing the bibliography, the corresponding coverage is 30.46%. These are local overlap measurements, not Turnitin similarity scores.

### Cited Findings
- The revised paper contains 5,685 normalized tokens and the earlier manuscript contains 5,482. The method lowercases text, removes punctuation, repairs PDF line-end hyphenation, and then marks every revised-paper token that belongs to an exact matching n-gram in the comparison file. With an 8-word threshold, 45.84% of the full revised document and 33.53% of its pre-reference body are covered. With both 12- and 20-word thresholds, coverage is 43.34% for the full document and 30.46% for the body. — [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- The bibliography contributes 1,052 consecutive matching tokens. It is unchanged, which explains 18.50 percentage points of full-document 12-word coverage (1,052 / 5,685). Bibliography matching is normally handled separately in similarity systems, so the body-only result is the more useful diagnostic here. — [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- The similarity-report PDF and AI-report PDF reproduce the same earlier manuscript pages. Comparing the revision with either report's embedded manuscript gives the same exact-coverage values as comparison with the standalone earlier PDF: 45.84% full-document coverage and 33.53% body coverage at 8 words, and 43.34% full-document coverage and 30.46% body coverage at 12 or 20 words. They therefore represent one prior text, not three independent sources. — [similarity report](<C:/Users/naresh v/Downloads/A Physics-Consistent Electrochemical.pdf>); [AI report](<C:/Users/naresh v/Downloads/A Physics-Consistent Electrochemical (1).pdf>); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- Section-level 12-word coverage against the earlier manuscript is: abstract/front matter 9.6%, introduction 0%, related-work narrative before the floated table 0%, governing physics 100%, proposed-model region 64.0%, experimental methodology 34.5%, results/discussion 3.3%, conclusion 0%, and acknowledgment 0%. PDF float ordering places Table I inside the extracted proposed-model region, so the 64.0% figure should not be interpreted as prose-only overlap. — [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)

### Inferences
- The revision successfully replaced most general narrative prose. The remaining local overlap is concentrated in formulas and their explanations, two tables, evaluation protocol text, front matter, and the bibliography.
- An external service that excludes the bibliography but compares against the institution's stored earlier submission may still report a material self-match because the governing-physics and evaluation-method passages remain verbatim.
- The identical results for the two reports and the standalone earlier manuscript should not be averaged or counted three times.

### Gaps
- This audit has no access to Turnitin's repository, matching algorithm, exclusions, source weighting, or AI-writing model. It cannot predict a Turnitin percentage or AI-writing indicator.
- PDF extraction can reorder two-column text and floated tables. The exact-token method is reproducible and conservative for verbatim reuse, but it is not a semantic plagiarism detector.

## Which retained material is structural, and which prose still presents a self-match risk?

### Takeaway
All 1,412 body tokens covered by exact matches of at least 12 words occur in eight continuous retained blocks. Several blocks are expected technical material, but four also contain full explanatory paragraphs that could be rewritten without changing equations or scientific meaning.

### Cited Findings
- **Author and institution block: 41 tokens.** Names, affiliations, and location are identical. This is required metadata and is not original scientific prose. — [revised LaTeX, author block](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex)
- **Index terms and section transition: 21 tokens.** Standard technical names such as physics-informed neural network, lithium-ion battery, SPMe, and diffusion-induced stress are unchanged. These terms should remain technically correct. — [revised LaTeX](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex)
- **Governing physics: 409 tokens.** The diffusion equation, boundary conditions, thermal balance, stress integral and stress equations are unchanged, as expected. However, their surrounding explanations are also verbatim, including “We represent one negative-electrode particle by a sphere…,” “PyBaMM returns one volume-averaged cell temperature…,” and the caution that the stresses are linear-elastic indicators rather than measured stress or crack simulation. The equations and citations can stay; the explanatory sentences are the strongest avoidable self-match in this section. — [revised LaTeX, Governing Physics](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L155); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- **Table I plus architecture/residual material: 522 tokens.** The ten-study comparison table is retained word for word. The same exact block continues into prose beginning “The following output forms satisfy both initial conditions exactly,” followed by hard-constraint and physics-residual explanations. Study names, tasks, and physics categories are factual, but the table's “key distinction” wording and the surrounding method prose can be rewritten. — [revised LaTeX, Table I and model](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L124); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- **Experimental-configuration table: 81 tokens.** Model type, parameter set, cutoff voltage, sample counts, network size, optimizer, training limits, seeds, and mechanical constants are unchanged. These values must remain stable for reproducibility; headings and caption can change, but changing numeric values merely to reduce overlap would be improper. — [revised LaTeX, configuration table](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L283)
- **Metric definitions and explanation: 140 tokens.** RMSE, relative L2 error, nine metric names, the reason MAPE is inappropriate for stress, temperature-rise scoring, and the mass-balance equation remain verbatim. Standard formulas should remain; the prose explaining metric choice can be recast in the authors' own words. — [revised LaTeX, Evaluation measures](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L315)
- **Stress-consistency, aggregation, and latency protocol: 167 tokens.** The paragraphs beginning “For the coupling check…” and “Runtime is measured…” remain verbatim, including the relative-change and speedup equations. These are project-specific prose and are the highest-priority text to rewrite after the governing-physics explanations. — [revised LaTeX, coupling and runtime](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L334)
- **Pending-results table: 31 tokens.** Column headings, model names, and “pending” cells are identical. This is structural content; the pending values correctly avoid inventing results. — [revised LaTeX, pending results](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L366)
- Outside those eight blocks, only 142 body tokens are covered by isolated exact 8–11 word matches, equal to 3.06% of the 4,635-token pre-reference body. No additional exact 12-word block remains. This residual is mostly short terminology and conventional transitions, but it should not be presented as a plagiarism percentage. — [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- The revised LaTeX contains the heading `\subsection{Evaluation measures}` twice in succession. This is an editing defect rather than an originality problem and should be removed in the next build. — [revised LaTeX](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.tex#L315)

### Inferences
- Required equations, parameter values, author names, model names, and citations should not be distorted for the sake of a lower automated score.
- The clearest legitimate reduction in self-overlap would come from rewriting the prose around the governing equations, hard constraints, evaluation metrics, coupling check, and timing protocol, and from independently redrafting Table I's comparison language.
- If the earlier submission is the authors' own institutional draft, repository exclusion is the accurate administrative remedy for self-match. Rewording can improve authorship clarity, but it cannot substitute for disclosure or repository handling.

### Gaps
- The reports do not reveal which individual retained passages would be counted after a new submission with bibliography, quotations, cited text, and small-match filters applied.
- This audit does not determine whether the earlier institutional submission can be excluded; that requires the instructor or Turnitin administrator.

## What can and cannot be concluded from this audit?

### Takeaway
The audit supports a concrete editing decision: retain the mathematics and reproducibility values, but redraft the accompanying explanations and comparison-table wording. It does not support a claim of “0% AI,” “under 10% Turnitin similarity,” or “plagiarism free.”

### Cited Findings
- The revised abstract, introduction, related-work prose, conclusion, and most results/limitations prose have no exact 12-word block from the earlier manuscript. This shows meaningful textual revision in the sections most often written as narrative. — [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- The body still contains 1,412 tokens in exact runs of at least 12 words, dominated by governing physics, Table I, equations and their explanations, metric descriptions, and timing protocol. A claim that the paper has negligible overlap would therefore be inaccurate unless the prior institutional submission and legitimate technical matter are explicitly excluded. — [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf); [earlier manuscript](<C:/Users/naresh v/Downloads/PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf>)
- The supplied AI report is a report about the earlier manuscript, not a test of the revised PDF. Its percentage cannot be transferred to the new document, and the local exact-match method does not estimate AI authorship. — [AI report](<C:/Users/naresh v/Downloads/A Physics-Consistent Electrochemical (1).pdf>); [Revised PDF](../../deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf)

### Inferences
- A responsible next revision should focus on authorship and clarity: each author should verify every paragraph, rewrite explanations in language they can defend orally, and preserve accurate citations and unchanged experiment settings.
- After that revision, a new institutional similarity report is still necessary. The settings used for bibliography, quoted material, cited text, small matches, and the earlier repository submission should be recorded beside the score.
- Any public plagiarism or AI-checking website would use a different corpus and opaque algorithm. Its number would not validate compliance with the institution's Turnitin configuration.

### Gaps
- No third-party checker was run as part of this local audit, so there is no external-database result for the revised PDF.
- There is no reliable local procedure for proving human authorship or guaranteeing a zero AI-writing score.
