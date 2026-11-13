from pathlib import Path


path = Path(r"E:\Naresh_Projects\PINN_EV_battery\paper_authorship_revision\main.tex")
text = path.read_text(encoding="utf-8")


def between(start: str, end: str, replacement: str) -> None:
    global text
    left = text.index(start)
    right = text.index(end, left)
    text = text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def exact(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match, found {count}: {old[:70]!r}")
    text = text.replace(old, new, 1)


exact(
    r"\title{{\headingfont\bfseries A Physics-Consistent Electrochemical--Thermal PINN with Concentration-Coupled Particle Stress for Lithium-Ion Battery Surrogate Modeling}}",
    r"\title{{\headingfont\bfseries A Concentration-Driven Stress Layer for a Physics-Informed Lithium-Ion Battery Discharge Surrogate}}",
)

between(
    r"\begin{abstract}",
    r"\end{abstract}",
    r"""\begin{abstract}
Our first battery PINN predicted concentration, temperature, and two particle-stress components as separate outputs. That arrangement allowed the stress curves to disagree with the concentration field that was supposed to generate them. We correct that weakness for constant-current discharge. The revised network learns negative-particle lithium concentration and the cell's volume-averaged temperature from radius, time, C-rate, and ambient temperature. Solid-diffusion, boundary-flux, and thermal-balance residuals are evaluated during training. Radial and tangential stresses are then obtained by integrating the predicted concentration through a spherical elastic-particle relation. PyBaMM SPMe simulations using the Chen2020 parameters provide the reference trajectories. Complete operating conditions, rather than shuffled rows, form the training, validation, and test sets. Data-only, conventional four-output PINN, and concentration-driven stress variants are compared at three label budgets and three random seeds. The evaluation covers field error, peak stress, particle mass balance, cross-field consistency, and inference time. The scope is a simulated discharge surrogate. The work does not estimate aging, state of health, remaining life, or experimentally measured stress.
""",
)

between(
    r"\section{Introduction}",
    r"\section{Related Work and Research Gap}",
    r"""\section{Introduction}
Current, voltage, and a few temperature sensors describe only part of what happens inside an electric-vehicle cell. They do not show the lithium gradient within an active-material particle, and that gradient is the quantity used to calculate diffusion-induced stress. A porous-electrode model can recover such internal states, but repeatedly solving it is inconvenient in studies that require thousands of evaluations. The Doyle--Fuller--Newman equations provide the detailed starting point \cite{doyle1993dfn}. Models such as the single-particle model with electrolyte dynamics (SPMe) reduce that cost while retaining important transport effects \cite{marquis2019spme,richardson2020gspm,planella2022continuum}. A PINN provides another route: simulated states guide the fit, and derivatives of the network are tested against selected equations \cite{raissi2019pinn}.

Battery PINNs have already been used for electrochemical surrogates, parameter estimation, temperature prediction, and health assessment \cite{hassanaly2024part1,hassanaly2024part2,wang2023temperature,wang2024parameter}. It would therefore be inaccurate to claim novelty simply because electrochemical, thermal, and mechanical quantities appear in one study. Wei \emph{et al.}, for instance, include all three domains together with aging mechanisms \cite{wei2026etm}. Our question is narrower and comes from a problem found in the earlier project model: if concentration and stress are learned independently, do the two predictions still describe one particle state?

In that earlier network, radial and tangential stress had their own output neurons. A mechanical penalty shaped those outputs, but it did not force them to equal the stress implied by the predicted concentration. A smooth stress curve could therefore score well against its labels and remain inconsistent with the electrochemical branch. The revised architecture removes both stress heads. It predicts concentration and temperature, then applies the spherical-particle equations of Cheng and Verbrugge to the concentration field \cite{cheng2009stress}.

The paper examines four concrete design choices. First, C-rate and ambient temperature are inputs so that one network covers the sampled discharge conditions. Second, stress is a deterministic post-processing layer rather than a learned target. Third, full simulation trajectories are assigned to separate data partitions; no row from a test condition is seen during fitting. Fourth, the same partitions and labels are used for three network variants, three label budgets, and three seeds. We judge the change by prediction error, conservation error, cross-field agreement, and timing. These tests concern single discharges only. A claim about degradation would require cycling data and health-related targets that are absent here.
""",
)

between(
    r"\section{Related Work and Research Gap}",
    r"\begin{table*}[t]",
    r"""\section{Related Work and Research Gap}
\subsection{Electrochemical surrogates}
Raissi \emph{et al.} introduced the now-standard combination of observations and differential-equation residuals \cite{raissi2019pinn}. For batteries, Hassanaly \emph{et al.} built PINN surrogates for single-particle and pseudo-two-dimensional models and studied parameter inference across fidelity levels \cite{hassanaly2024part1,hassanaly2024part2}. Their work is especially relevant to our concentration branch and to the use of conservation terms when labels are limited. J. Wang \emph{et al.} estimated electrochemical parameters with parallel physics-informed branches \cite{wang2024parameter}, while Zhuang \emph{et al.} examined SPM responses under varying currents with PINNs and a Fourier-enhanced operator network \cite{zhuang2026spm}. These studies establish the electrochemical surrogate problem. They do not test whether a predicted particle-stress field agrees with the model's own concentration.

\subsection{Thermal, health, and system-level studies}
Recent papers attach physics at different places in the learning pipeline. Y. Wang \emph{et al.} use electrical and heat-generation relations in a recurrent temperature predictor \cite{wang2023temperature}. Fu \emph{et al.} combine an equivalent-circuit description with recurrent learning for a spacecraft battery, and Dang \emph{et al.} constrain a state-of-charge estimator with a differential equation \cite{fu2024spacecraft,dang2024soc}. Jiang \emph{et al.} address the spatial heat map of a battery pack rather than particle-scale fields \cite{jiang2025heatmap}. Health studies use transfer learning, chosen physical features, small cycling datasets, or coupled aging mechanisms \cite{ye2024soh,tian2025generic,xiong2025diagnostics,chen2025smallsample,luo2025hybrid}. Table~\ref{tab:gap} separates these tasks because the phrase ``physics-informed'' alone does not reveal what the network predicts or what data support its claims.

\subsection{Why mechanics remains in the model}
Electrochemical--thermal--mechanical coupling is not new. Lyu \emph{et al.} include these components in a battery digital twin \cite{lyu2024digitaltwin}, and classical insertion-particle analyses relate lithium gradients to radial and circumferential stress \cite{cheng2009stress,christensen2006fracture,xu2016electrochemomechanics}. Our change lies in the direction of that link: the concentration prediction determines stress, so the two fields cannot drift apart during inference.

We retained mechanics because it can be evaluated from each discharge trajectory. Degradation is a different target. SEI growth, plated lithium, crack evolution, capacity loss, and state of health develop over repeated cycles and require their own parameters and observations. Adding one degradation equation to the current single-discharge dataset would not validate an aging model. Mechanics and degradation can coexist in later work, with stress serving as one input to a cycle-level model, but replacing mechanics now would leave the present data unable to support the new claim.
""",
)

exact(
    r"\caption{Task, physics representation, and unresolved point in ten related battery studies}",
    r"\caption{How ten battery-learning studies relate to the present concentration--stress experiment}",
)

exact(
    r"\caption{Proposed \modelname{} architecture. The temperature branch matches the lumped thermal assumption, while stress is an analytical function of predicted concentration.}",
    r"\caption{Computation used in the revised surrogate. Radius enters the concentration branch; the lumped-temperature branch has no radial coordinate. Both stress components are calculated from the concentration grid.}",
)

between(
    r"\subsection{Reproducible implementation}",
    r"\section{Experimental Methodology}",
    r"""\subsection{Implementation sequence}
Each C-rate and ambient-temperature pair is solved as one PyBaMM simulation. We export physical time, negative-electrode interfacial current, heat generation, average cell temperature, and the particle-concentration grid. The data pass through five steps. Complete conditions are assigned to training, validation, or test subsets. Scaling constants are then calculated from training conditions only, while unscaled variables are kept for the residual calculations. Labeled rows and collocation points are drawn independently from the training trajectories. Adam updates the network, and the state with the lowest validation objective is saved under the stated patience rule. Finally, that saved state is evaluated once on the untouched test conditions, after which stress and all metrics are calculated.

For a fixed seed, every model sees the same condition groups and labeled indices. Collocation points change during training, but the validation and test grids stay fixed. The run record stores the software version, PyBaMM parameter set, seed, label budget, selected epoch, elapsed time, and number of trainable parameters.

Stress post-processing is performed separately for every time and operating condition. Concentration values are sorted by particle radius and integrated with a cumulative trapezoidal rule. The analytic limit is used at the centre. This detail matters: inserting a small artificial denominator in place of $r^3$ changes the centre value and can move the reported stress peak.

""",
)

exact(
    r"\caption{Operating conditions and matched comparisons. Model variants differ in formulation, while partitions, label counts, and random seeds remain fixed.}",
    r"\caption{Condition-level split and matched training runs. A change of model does not change the selected labels, seed, or held-out trajectories.}",
)

between(
    r"\section{Experimental Methodology}",
    r"\begin{figure*}[!b]",
    r"""\section{Experimental Methodology}
\subsection{Reference trajectories and data split}
The reference cases are constant-current SPMe discharges with lumped thermal coupling and the Chen2020 parameter set. Simulation stops when terminal voltage reaches 2.5~V. From each run we retain 100 time instants and 25 particle-radius locations. Twelve training conditions combine C-rates $\{0.5,1.0,1.5,2.0\}$ with ambient temperatures $\{10,25,35\}^{\circ}$C. Four validation conditions use $\{0.75,1.75\}$C at $\{15,30\}^{\circ}$C. The final tests are 1.25C at $20^{\circ}$C and $32.5^{\circ}$C. The 18 trajectories contain 45,000 radius--time records.

The split is made before row sampling. Thus, all radii and times from a condition remain together. Validation conditions control checkpoint selection and early stopping; neither test condition is used until fitting is complete. Figure~\ref{fig:design} records the operating points and the repeated comparisons.

""",
)

between(
    r"\subsection{Models and optimization settings}",
    r"\begin{table}[t]",
    r"""\subsection{Models and training settings}
We train three versions of the surrogate. The \textbf{data-only} version follows the earlier design and directly predicts concentration, temperature, radial stress, and tangential stress. The \textbf{four-output PINN} uses the same outputs and adds diffusion, thermal, boundary, and mechanical-equilibrium penalties. The \textbf{revised \modelname} predicts only concentration and temperature; its stresses come from Eqs.~\eqref{eq:A}--\eqref{eq:st}.

Each training trajectory contributes 50, 150, or 300 labeled records. Seeds 11, 22, and 33 produce 27 main runs. Adam begins with a learning rate of $10^{-3}$, and the gradient norm is limited to 10. A run stops after 2500 epochs or after 350 epochs without validation improvement. For every training condition, 900 interior points and 120 boundary points are used for the physics terms. Evaluation always reloads the checkpoint chosen on validation data.

""",
)

exact(r"\caption{Full Experimental Configuration}", r"\caption{Simulation and training settings used in every matched run}")

between(
    r"\subsection{Controlled comparisons}",
    r"\subsection{Evaluation measures}",
    r"""\subsection{What each comparison isolates}
The data-only and four-output PINN pair asks whether residual terms improve the original architecture when labels are sparse. The four-output PINN and \modelname{} pair evaluates the complete revision: stress heads are removed and radius is also removed from the lumped-temperature branch. This second comparison should therefore be read as an architecture comparison, not as proof that one individual change caused the full difference. Repeating the runs at three label budgets shows whether either formulation is especially sensitive to limited supervision.

Three optional checks can narrow the diagnosis if computing time allows. We can remove only the thermal residual, apply the elastic operator to concentration from the data-only network, or replace only the two stress heads while leaving the former temperature inputs unchanged. These checks are secondary to the predeclared 27-run experiment.

\subsection{Evaluation measures}
""",
)

exact(
    "One score cannot describe a coupled field surrogate. We therefore report nine measures spanning prediction, physical checks, and runtime.",
    "A low average error can hide a poor boundary value, a missed stress peak, or an inconsistent pair of fields. We therefore use nine measures that cover prediction, physical checks, and runtime.",
)

exact(
    r"\caption{Timing procedure for the PINN and SPMe reference solver on one held-out query batch.}",
    r"\caption{Latency measurement applied to the same held-out query batch for the trained network and the SPMe solver.}",
)

between(
    r"\subsection{Aggregation and decision rule}",
    r"\section{Results and Discussion}",
    r"""\subsection{Aggregation and acceptance rule}
We first calculate every metric for one seed and one test condition. The two conditions are averaged within a seed; the table then gives the mean and sample standard deviation across the three seeds. We also retain the worse condition because the mean could conceal a temperature-dependent failure. Three seeds are enough to show visible run-to-run spread, but not enough for a strong significance claim.

The revised design is accepted only if it removes the constructed concentration--stress mismatch, keeps stress error from materially increasing, and preserves concentration and temperature accuracy across the three label budgets. A coupling residual of zero is expected by construction and is not sufficient evidence on its own. Speed comparisons are valid only for the fixed batch, hardware, precision, warm-up, and synchronization procedure in Fig.~\ref{fig:latency_protocol}. Training duration and parameter count are reported separately.

""",
)

between(
    r"\section{Results and Discussion}",
    r"\section{Limitations and Threats to Validity}",
    r"""\section{Results and Discussion}
\subsection{Current status of the result table}
Table~\ref{tab:results} is connected to two notebook outputs: \texttt{paper\_metrics\_summary.csv} and \texttt{variation\_gap\_vs\_previous.csv}. The table still shows placeholders because the full three-seed run has not been copied into this draft. Before submission, the notebook must run with \texttt{quick\_mode=False}, all 27 runs must complete, and the macros at the beginning of this file must be filled from the exported summaries. We do not estimate missing numbers from a short run or from the expected physical trend.

\begin{table*}[t]
\centering
\caption{Reserved test conditions at 300 labels per training condition; entries remain pending until all three seeds finish}
\label{tab:results}
\footnotesize
\resizebox{0.98\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Model & Concentration Rel-$L_2$ & Temp.-rise RMSE [K] & Tangential-stress RMSE [MPa] & Mass-balance RMSE [mol m$^{-3}$ s$^{-1}$] & Median speedup [$\times$] \\
\midrule
Data-only & \DataC & \DataT & \DataStress & \DataMass & \DataSpeed \\
Four-output PINN & \PriorC & \PriorT & \PriorStress & \PriorMass & \PriorSpeed \\
Revised \modelname & \ProposedC & \ProposedT & \ProposedStress & \ProposedMass & \ProposedSpeed \\
\bottomrule
\end{tabular}
}
\end{table*}

\subsection{Plots retained with the metrics}
The notebook also exports label-budget curves and held-out trajectories for surface concentration, temperature rise, and the two stresses. Particle mass-balance residual appears in a separate panel. Every graphic identifies its test condition and states how seeds were combined. The scalar table supports quick comparison; the curves show boundary offsets, timing errors, and brief stress peaks that disappear in an average.

All summaries originate from one condition-level results file. Before making a table, the script checks for the expected 27 runs, disjoint condition groups, and three seeds in every aggregate. If training fails, that run is marked incomplete. It is not silently removed to improve the average.

\subsection{Interpretation planned before seeing the numbers}
The first model pair determines whether equation residuals help the original four-output network at low label counts. The second determines whether enforcing the concentration-to-stress construction improves cross-field consistency while retaining useful accuracy in concentration, temperature, and stress. Both held-out temperatures, every seed, field-level stress error, peak-stress error, and the independent mass-balance calculation must be considered together. Since the test points lie inside the sampled C-rate and temperature envelope, the experiment measures interpolation. It cannot support conclusions about charging, drive cycles, other chemistries, or extreme temperatures.

The curves also have basic physical checks. Surface reaction during discharge can produce a radial lithium gradient and hence diffusion-induced stress. Equations~\eqref{eq:sr} and \eqref{eq:st} return zero stress for a uniform concentration. Higher discharge rate is expected to produce a larger transient gradient. The thermal curve should reflect the competition between generated heat and convection in Eq.~\eqref{eq:thermal}. These expectations help identify implementation errors, but they do not replace measured comparisons.

""",
)

between(
    r"\section{Limitations and Threats to Validity}",
    r"\balance",
    r"""\section{Limitations and Threats to Validity}
The reference values are simulation outputs. Reproducing PyBaMM means that the network approximates this SPMe setup; it does not validate the hidden fields against a laboratory cell. The spatial model is one negative-electrode particle, and temperature is one cell average. Material properties are fixed. Through-thickness temperature, electrode-scale stress, plasticity, binder response, fracture, and feedback from stress to diffusion are outside the equations used here.

Young's modulus, Poisson's ratio, and partial molar volume are literature values rather than measurements from the particular LG M50 cell represented by Chen2020. Stress is consequently a calculated indicator. The Chen2020 set was assembled from multiscale experiments \cite{chen2020parameterization}, but using that set does not establish the exact internal properties of another cell.

The heat-generation signal also comes from the reference SPMe run. Training covers constant current only. An arbitrary drive-cycle current may require a different input representation and additional tests. Optimization brings its own uncertainty because data terms and physical residuals can have very different gradient scales. Repeated seeds and off-grid residual checks can reveal unstable solutions, but they cannot repair an incomplete governing model.

This paper does not treat stress as a synonym for degradation. PyBaMM includes models for SEI growth, plating, loss of active material, and cracking \cite{sulzer2021pybamm}; testing them would require cycling schedules, degradation parameters, capacity or health targets, and held-out cells or profiles. A future aging model may use the present stress indicator, but the current evidence stops at one discharge.

\subsection{Parameter and numerical sensitivity}
The elastic operator scales with $E\Omega/(1-\nu)$. A mechanical error can therefore arise from $E$ or $\Omega$ even when concentration is predicted correctly. Our sensitivity check reuses each stored concentration field and changes $E$ and $\Omega$ separately by $\pm20\%$, without retraining. The resulting change in peak stress is reported. Conclusions about diffusion remain tied to the fixed Chen2020 value of $D_s$.

We also repeat radial integration with 25, 50, and 100 points. Peak stress and mass-balance error should settle as the grid is refined. If they do not, the apparent model difference may be a quadrature effect.

\subsection{Reading failed runs}
Training histories help locate the source of failure. A small diffusion residual combined with a large concentration error suggests a boundary, scaling, or initial-state problem. A close fit to labels and a large off-grid residual indicates weak physical generalization. Accurate concentration with inaccurate stress points to the elastic constants or the radial integral. A nearly constant temperature offset suggests heat capacity or heat generation; an offset that changes with ambient temperature directs attention to convection.

Non-finite objectives, checkpoints selected unusually early, and runs still improving at epoch 2500 remain in the audit record with their seeds. Further robustness work would need noise, parameter shifts, and time-varying current. Agreement with an imposed equation shows internal consistency with that equation, not completeness of the battery physics.

""",
)

between(
    r"\section{Conclusion}",
    r"\section*{Data and Code Availability}",
    r"""\section{Conclusion}
We changed the original four-output battery PINN after finding that its stress heads could disagree with its concentration prediction. The revised network learns particle concentration and lumped temperature; a spherical elastic-particle calculation turns the concentration grid into radial and tangential stress. Diffusion, boundary, and energy-balance terms constrain the two learned fields. Entire operating conditions are held out, and the comparison repeats the same labels and seeds for data-only, four-output PINN, and revised models.

The architectural constraint guarantees internal concentration--stress agreement, but that identity alone is not a performance result. The completed experiment must show that concentration, temperature, and stress errors remain acceptable at all label budgets. If stress consistency improves while field accuracy deteriorates, the correct conclusion will be that the constraint exposed a concentration-learning or parameter-calibration problem. Laboratory measurements, dynamic-current tests, and cycling data are still required before the model can be used to make claims about cell degradation or health.

""",
)

between(
    r"\section*{Data and Code Availability}",
    r"\section*{Acknowledgment}",
    r"""\section*{Data and Code Availability}
The accompanying notebook creates the PyBaMM trajectories, performs the 27 matched runs, and exports checkpoints, condition-level metrics, and label-budget summaries. A reproducible release should preserve the generated data, trained weights, software versions, and seeds recorded in Table~\ref{tab:config}.

""",
)

exact(
    "The authors acknowledge the Department of Computer Science and Engineering, KPR Institute of Engineering and Technology, for supporting this project.",
    "The authors thank the Department of Computer Science and Engineering at KPR Institute of Engineering and Technology for the facilities used in this project.",
)

# Keep terminology consistent after the rewritten comparisons.
text = text.replace("Prior-style PINN", "Four-output PINN")
text = text.replace("Proposed \\modelname", "Revised \\modelname")

path.write_text(text, encoding="utf-8")
print(path)
print(f"characters={len(text)}")
