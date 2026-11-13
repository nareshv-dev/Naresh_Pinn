# Public-web phrase-overlap audit of the revised PINN battery manuscript

## Does the manuscript contain distinctive prose that appears elsewhere on the public web?

### Takeaway
Fourteen exact-phrase probes sampled from the abstract, introduction, related work, architecture, evaluation, results, limitations, and conclusion produced no credible exact match to another public document. On this limited public-web audit, the manuscript's project-specific prose presents a low plagiarism risk; the search cannot measure Turnitin's percentage or inspect private institutional repositories.

### Cited Findings
- **Abstract, line 92 — low risk.** Searches for the distinctive sentence beginning “That arrangement allowed the stress curves to disagree...” returned no relevant exact match. The unrelated results did not contain the manuscript wording. Search record: [public-web query](https://www.google.com/search?q=%22That+arrangement+allowed+the+stress+curves+to+disagree+with+the+concentration+field%22).
- **Introduction, line 105 — low risk.** The sentence beginning “A smooth stress curve could therefore score well...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22A+smooth+stress+curve+could+therefore+score+well+against+its+labels%22).
- **Related work, line 119 — low risk.** The sentence beginning “Adding one degradation equation to the current single-discharge dataset...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22Adding+one+degradation+equation+to+the+current+single-discharge+dataset%22).
- **Architecture, line 199 — low risk.** The wording “A trajectory-specific end time is deliberately avoided...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22A+trajectory-specific+end+time+is+deliberately+avoided%22).
- **Evaluation, line 318 — low risk.** The sentence beginning “A low average error can hide a poor boundary value...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22A+low+average+error+can+hide+a+poor+boundary+value%22).
- **Evaluation, line 325 — low risk.** The phrase “Temperature scores use rise above ambient...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22Temperature+scores+use+rise+above+ambient%22).
- **Acceptance rule, line 360 — low risk.** The sentence beginning “A coupling residual of zero is expected by construction...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22A+coupling+residual+of+zero+is+expected+by+construction%22).
- **Results status, line 364 — low risk.** The sentence “We do not estimate missing numbers from a short run...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22We+do+not+estimate+missing+numbers+from+a+short+run%22).
- **Results plots, line 385 — low risk.** The wording “Particle mass-balance residual appears in a separate panel” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22Particle+mass-balance+residual+appears+in+a+separate+panel%22).
- **Limitations, line 397 — low risk.** The sentence “Stress is consequently a calculated indicator” produced no battery-related exact match. Search record: [public-web query](https://www.google.com/search?q=%22stress+is+consequently+a+calculated+indicator%22+battery).
- **Sensitivity, line 404 — low risk.** The phrase “The elastic operator scales with” plus “partial molar volume” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22The+elastic+operator+scales+with%22+%22partial+molar+volume%22).
- **Validity discussion, line 411 — low risk.** The sentence beginning “Agreement with an imposed equation shows internal consistency...” produced no relevant exact match. Search record: [public-web query](https://www.google.com/search?q=%22Agreement+with+an+imposed+equation+shows+internal+consistency%22).
- **Conclusion, line 415 — low risk.** Searches for “stress heads could disagree with its concentration prediction” and “The revised network learns particle concentration and lumped temperature” produced no relevant exact matches. Search records: [query 1](https://www.google.com/search?q=%22stress+heads+could+disagree+with+its+concentration+prediction%22), [query 2](https://www.google.com/search?q=%22The+revised+network+learns+particle+concentration+and+lumped+temperature%22).
- **Title — low risk.** The exact distinctive title stem “A Concentration-Driven Stress Layer” did not return a matching public paper. Search record: [public-web query](https://www.google.com/search?q=%22A+Concentration-Driven+Stress+Layer%22+battery+PINN).

### Inferences
- The searched narrative appears independently worded. The strongest evidence is the absence of matches for long, project-specific strings that combine architecture choices, validation limitations, and experimental procedure.
- The conclusion repeats ideas and some terminology from the abstract and introduction. That is internal repetition, not evidence of copying from another source, although a similarity engine may still highlight repeated wording within the submission depending on its settings.
- No responsible percentage can be derived from fourteen search probes. A public search engine does not index all publisher text, student submissions, subscription databases, or Turnitin's repository.

### Gaps
- Search-engine non-results do not prove that no matching text exists in unindexed or private sources.
- This audit cannot see the KPR institutional repository match that previously contributed 17 percentage points unless the underlying submission is publicly indexed.
- No public tool can reproduce Turnitin's source corpus or its exclusion settings, so this audit cannot certify “under 10%.”

## Which overlaps are expected technical or bibliographic language rather than suspicious copying?

### Takeaway
The material public-web matches are confined to article titles, model names, standard equations, and short technical descriptors that the manuscript attributes with citations. These are expected overlaps. The prose summaries around them are substantially different from the source abstracts reviewed.

### Cited Findings
- **Bibliography and Related Work — expected, very low risk.** The full title “PINN surrogate of Li-ion battery models for parameter inference, Part I: Implementation and multi-fidelity hierarchies for the single-particle model” necessarily matches the cited publication title. The source describes replacing SPM/P2D solvers with a PINN surrogate and multi-fidelity training; the manuscript paraphrases this in its own words and cites it. [Hassanaly et al., arXiv](https://arxiv.org/abs/2312.17329).
- **Bibliography/Table I — expected, very low risk.** The title “Temperature state prediction for lithium-ion batteries based on improved physics informed neural networks” and terms such as electrical model, heat-generation model, and thermal model occur in the source. The manuscript gives a short task-focused summary and cites the paper; it does not reproduce the abstract's sentences. [Journal of Energy Storage article](https://www.sciencedirect.com/science/article/pii/S2352152X23022600).
- **Bibliography/Table I — expected, very low risk.** “Digital twin modeling method for lithium-ion batteries based on data-mechanism fusion driving” is the cited title. The source explicitly describes an electrochemical--thermal--mechanical coupled model and particle-level stress. The manuscript uses those unavoidable domain labels in a compact comparative table with a citation. [Green Energy and Intelligent Transportation article](https://www.sciencedirect.com/science/article/pii/S2773153724000148).
- **Parameter-set description — expected, very low risk.** The manuscript names the Chen2020 LG M50 parameter set and states that it was assembled using multiscale experiments. The original paper reports parameterization of a cylindrical LG M50 21700 cell and the use of those parameters in a P2D model. The manuscript cites the source and does not copy its methodological description. [Journal of The Electrochemical Society article](https://doi.org/10.1149/1945-7111/AB9050).
- **Equations and nomenclature — expected, low risk.** Spherical Fickian diffusion, a lumped heat balance, the Cheng--Verbrugge stress relations, RMSE, relative L2 error, and mass balance have limited mathematical forms. Their symbols and surrounding definitions may resemble cited literature even when independently typeset. This is normal technical overlap when the source is cited; the relevant stress source is identified in the manuscript as Cheng and Verbrugge. [Journal of Power Sources DOI](https://doi.org/10.1016/j.jpowsour.2009.01.021).
- **Reference list — expected, very low risk.** Every article title, author list, journal, DOI, and pagination will match bibliographic databases verbatim. Similarity systems normally offer bibliography exclusion; these strings should not be paraphrased.

### Inferences
- If a checker highlights the literature table, the likely cause is the necessary reuse of paper titles and compact domain labels such as “multi-fidelity hierarchy” or “electrochemical--thermal--mechanical coupling,” not copied argumentative prose.
- Rewriting mathematical laws or article titles to lower a numerical score would reduce clarity and citation accuracy. Configure the similarity report to exclude bibliography, quoted material, and small matches before judging the prose.
- The most plausible remaining source of a score above 10% is self-overlap with the earlier KPR submission, because the public-web probe found no comparable prose source and the prior report attributed 17% to that repository.

### Gaps
- The audit did not compare every sentence against the full text of all 26 cited papers; several publisher full texts are access-controlled.
- Equation-by-equation provenance was not re-verified because the task was a phrase-overlap audit, not a scientific citation audit.
- Only the institution or Turnitin administrator can exclude an earlier repository submission or confirm whether bibliography/small-match exclusions are permitted for the final assessment.

## What should be reported to the authors from this audit?

### Takeaway
No suspicious public-web prose match was found in the sampled passages. The defensible statement is that the manuscript passed a targeted public-web phrase audit; it is not a plagiarism certificate, an AI-writing score, or a prediction of Turnitin's final percentage.

### Cited Findings
- Google and other public search indexes cover public pages, whereas Turnitin states that its Similarity Report compares submissions against internet content, publications, and previously submitted student papers; the student-paper repository is not reproducible through ordinary web search. [Turnitin Similarity overview](https://guides.turnitin.com/hc/en-us/articles/234264354408?tab=article).
- Turnitin explains that a similarity score is the proportion of matching text, not itself a finding of plagiarism; academic judgment is required to interpret the highlighted sources. [Turnitin similarity-score guidance](https://guides.turnitin.com/hc/en-us/articles/234264354408?tab=article).
- The public source pages checked for related-work terminology confirm the paper titles and technical descriptors that are already cited, rather than exposing copied project-specific prose. [Hassanaly et al.](https://arxiv.org/abs/2312.17329); [Wang et al.](https://www.sciencedirect.com/science/article/pii/S2352152X23022600); [Lyu et al.](https://www.sciencedirect.com/science/article/pii/S2773153724000148); [Chen et al.](https://doi.org/10.1149/1945-7111/AB9050).

### Inferences
- The authors can submit this revised version for a fresh institutional check with reasonable confidence that distinctive prose is not readily traceable to public web pages.
- Before interpreting the new percentage, the guide should apply the venue's accepted exclusions and inspect whether the earlier KPR submission is being counted as a source.
- If any new report flags a passage, revision should be based on the actual matched source and citation need, not on arbitrary synonym replacement.

### Gaps
- A fresh Turnitin report is still required to know the score under the college's repository and exclusion settings.
- This audit cannot determine whether prose was produced by a person or a language model. “AI detection” is a separate probabilistic classification problem and is not measured by phrase search.
