from pathlib import Path


path = Path(r"E:\Naresh_Projects\PINN_EV_battery\paper_preview\battery_pinn_ieee_conference_originality_revision.tex")
text = path.read_text(encoding="utf-8")


def replace_between(start: str, end: str, replacement: str) -> None:
    global text
    left = text.find(start)
    if left < 0:
        raise RuntimeError(f"Start marker not found: {start}")
    right = text.find(end, left + len(start))
    if right < 0:
        raise RuntimeError(f"End marker not found: {end}")
    text = text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def replace_exact(old: str, new: str) -> None:
    global text
    if old not in text:
        raise RuntimeError(f"Exact text not found: {old[:80]}")
    text = text.replace(old, new, 1)


replace_between(
    r"\begin{abstract}",
    r"\end{abstract}",
    r"""\begin{abstract}
Battery surrogates are useful only when their coupled outputs describe the same physical state. A network may fit concentration, temperature, and stress labels separately while violating the concentration--stress relation inside an electrode particle. This work examines that specific failure mode for constant-current discharge. The proposed physics-consistent multiphysics PINN (\modelname) learns negative-particle lithium concentration and a volume-averaged cell temperature as functions of position, time, C-rate, and ambient temperature. Its objective includes solid-diffusion, symmetry, surface-flux, and thermal-energy residuals. Particle stresses are then calculated from the predicted concentration using a spherical linear-elastic operator; they are not independent neural outputs. Reference fields come from PyBaMM SPMe simulations with the Chen2020 parameter set. Evaluation is organized by complete operating conditions and compares three model variants at three label budgets and three random seeds. The reported measures examine prediction error, conservation, peak stress, agreement between electrochemical and mechanical fields, and inference time. The study concerns a discharge surrogate built from simulated data. It does not claim experimental stress validation, cycle aging, state-of-health estimation, or remaining-life prediction.""",
)

replace_between(
    r"\section{Introduction}",
    r"\section{Related Work and Research Gap}",
    r"""\section{Introduction}
An electric-vehicle battery-management system measures current, terminal voltage, and a limited number of temperatures. It cannot directly observe the lithium profile within an active-material particle or the stress caused by that profile. Electrochemical models can estimate these internal variables, although repeated numerical solves become costly when a controller, optimizer, or uncertainty study needs many queries. The Doyle--Fuller--Newman model gives a detailed porous-electrode description \cite{doyle1993dfn}. Reduced formulations, including the single-particle model with electrolyte dynamics (SPMe), trade some spatial detail for lower computational cost \cite{marquis2019spme,richardson2020gspm,planella2022continuum}. A physics-informed neural network offers a different approximation: it fits simulated states while its derivatives are checked against selected governing equations \cite{raissi2019pinn}.

Published battery PINNs already address parameter recovery, electrochemical surrogates, temperature prediction, and health estimation \cite{hassanaly2024part1,hassanaly2024part2,wang2023temperature,wang2024parameter}. Electrochemical, thermal, and mechanical terms in one model are therefore not sufficient evidence of novelty. Wei \emph{et al.}, for example, combine those domains with aging mechanisms for state-of-health estimation \cite{wei2026etm}. The present study asks a smaller question that can be tested with the available discharge data: does a stress field become more credible when the architecture forces it to come from the predicted concentration?

The earlier project model treated radial and tangential stress as trainable outputs. A mechanical residual restricted their shapes but did not make them unique functions of the network's concentration field. Consequently, a fitted stress curve could be acceptable on its own and still be inconsistent with the electrochemical prediction. In the revised model, the network predicts concentration and temperature only. The Cheng--Verbrugge spherical-particle relation converts concentration to stress \cite{cheng2009stress}. This removes one source of cross-field disagreement rather than adding another output head.

The study makes four testable contributions:
\begin{enumerate}
    \item a discharge PINN parameterized by C-rate and ambient temperature, with separate concentration and lumped-temperature branches;
    \item an analytical concentration-to-stress layer that eliminates independently fitted mechanical outputs;
    \item condition-level data partitions and validation-based checkpoint selection, so rows from one simulated trajectory cannot appear in both training and testing; and
    \item a matched comparison of three model variants over three label budgets and three seeds, assessed with prediction, conservation, coupling, and timing measures.
\end{enumerate}
The claims remain limited to single-discharge behavior. Cycle-dependent degradation requires different targets and a different dataset.""",
)

replace_between(
    r"\section{Related Work and Research Gap}",
    r"\begin{table*}[t]",
    r"""\section{Related Work and Research Gap}
\subsection{Electrochemical surrogate models}
The original PINN formulation joins observations with differential-equation residuals during training \cite{raissi2019pinn}. Hassanaly \emph{et al.} adapted this idea to single-particle and pseudo-two-dimensional battery models and studied parameter inference at multiple fidelity levels \cite{hassanaly2024part1,hassanaly2024part2}. Their results also show why conservation terms and staged training matter when labeled states are sparse. J. Wang \emph{et al.} used parallel physics-informed branches to estimate electrochemical parameters across operating regimes \cite{wang2024parameter}. Zhuang \emph{et al.} compared conventional PINNs with a Fourier-enhanced operator model for SPM responses under changing current \cite{zhuang2026spm}. These papers are the closest electrochemical references, but none evaluates a particle-stress field constructed from its own predicted concentration.

\subsection{Recent battery applications}
Physics enters battery-learning systems in several ways. Fu \emph{et al.} placed an equivalent-circuit relation inside a recurrent spacecraft-battery model, whereas Dang \emph{et al.} used a differential-equation constraint for state-of-charge estimation \cite{fu2024spacecraft,dang2024soc}. Jiang \emph{et al.} combined fidelity levels for spatial pack-temperature prediction \cite{jiang2025heatmap}. Other recent studies focus on state of health and use transfer learning, physically chosen features, limited-sample aging data, or reaction and mechanics priors \cite{ye2024soh,tian2025generic,xiong2025diagnostics,chen2025smallsample,luo2025hybrid}. Because these implementations use the word ``physics-informed'' for different mechanisms, Table~\ref{tab:gap} records both the task and the point at which physics enters the model.

\subsection{Thermal and mechanical coupling}
Y. Wang \emph{et al.} connected electrical and thermal relations to a recurrent predictor for cell temperature over life-cycle data \cite{wang2023temperature}. Lyu \emph{et al.} used electrochemical, thermal, and particle-mechanics components in a battery digital twin \cite{lyu2024digitaltwin}. Thus, the three-domain combination predates this paper. The distinction studied here is architectural: stress is recomputed from concentration instead of learned in parallel. The required elastic relation follows spherical insertion-particle analyses \cite{cheng2009stress,christensen2006fracture}. Lithium insertion changes particle volume, and a nonuniform concentration then produces different radial and circumferential stresses \cite{xu2016electrochemomechanics}. A uniform profile produces no diffusion-induced stress in this model.

\subsection{Reason for retaining mechanics}
Mechanical response and degradation describe different time scales. The stress considered here is an instantaneous response to a concentration gradient during a discharge. Aging models track irreversible change across cycles and need variables such as capacity, cycle count, SEI thickness, plated lithium, crack growth, or loss of active material. Those mechanisms can later be coupled to stress, but the present simulations do not contain the labels needed to validate them. Replacing the mechanical calculation with a degradation equation would therefore change the research problem without supplying the required evidence. Table~\ref{tab:gap} states this scope explicitly.""",
)

replace_exact(
    r"\caption{Positioning Against Ten Recent Physics-Informed Battery Studies}",
    r"\caption{Task, physics representation, and unresolved point in ten related battery studies}",
)

replacements = {
    "The negative active-material particle is treated as a sphere of radius $R_p$ with radial coordinate $r$. Under spherical symmetry and constant solid diffusivity $D_s$, lithium concentration $c_s(r,t)$ satisfies Fickian diffusion:":
        "We represent one negative-electrode particle by a sphere with radius $R_p$. With radial symmetry and a constant solid diffusivity $D_s$, its lithium concentration $c_s(r,t)$ follows",
    "The center-symmetry, interfacial-flux, and initial conditions are":
        "The equation is completed by symmetry at the center, the reaction flux at the surface, and the initial concentration:",
    "where $j_n$ is the negative-electrode interfacial current density and $F$ is Faraday's constant. The SPMe reference is generated in PyBaMM, an open-source battery modeling framework \\cite{sulzer2021pybamm}, using the Chen2020 LG M50 parameterization \\cite{chen2020parameterization}.":
        "Here $j_n$ denotes interfacial current density at the negative electrode and $F$ denotes Faraday's constant. PyBaMM supplies the SPMe solution used as the numerical reference \\cite{sulzer2021pybamm}; all cell parameters are taken from the Chen2020 LG M50 set \\cite{chen2020parameterization}.",
    "The reference simulation uses a volume-averaged cell temperature. Battery energy balances connect heat generation and cooling \\cite{bernardi1985energy}. The compatible lumped energy balance is":
        "PyBaMM returns one volume-averaged cell temperature for this configuration. We therefore use a lumped balance between internal heat release and cooling to the surroundings \\cite{bernardi1985energy}:",
    "where $q_{\\mathrm{gen}}$ is volumetric heat generation, $h$ is the heat-transfer coefficient, and $A_{\\mathrm{cool}}/V_{\\mathrm{cell}}$ is the cooling-area-to-volume ratio. Since $T$ is lumped, the proposed temperature network does not receive $r$; allowing a radial input would introduce a degree of freedom that the assumed physics cannot identify.":
        "In Eq.~\\eqref{eq:thermal}, $q_{\\mathrm{gen}}$ is heat generation per unit volume, $h$ is the convection coefficient, and $A_{\\mathrm{cool}}/V_{\\mathrm{cell}}$ converts surface cooling to a volumetric term. The temperature subnetwork has no radial input because the reference temperature has no radial resolution.",
    "where $\\Omega$ is partial molar volume, $E$ is Young's modulus, and $\\nu$ is Poisson's ratio. The radial and tangential stresses are computed as \\cite{cheng2009stress}":
        "The constants $\\Omega$, $E$, and $\\nu$ denote partial molar volume, Young's modulus, and Poisson's ratio. The resulting radial and circumferential stresses are \\cite{cheng2009stress}",
    "Equations~\\eqref{eq:A}--\\eqref{eq:st} are applied to the predicted concentration. The stress field is therefore mechanically interpretable but remains an analytical, linear-elastic estimate; it is not an experimental stress measurement or a crack-evolution model.":
        "During evaluation, Eqs.~\\eqref{eq:A}--\\eqref{eq:st} receive the network concentration rather than a separate stress prediction. These values are linear-elastic indicators. They should not be read as measured stresses or as a simulation of crack initiation and growth.",
    "Figure~\\ref{fig:architecture} shows the model. Inputs are normalized radius $\\bar r=r/R_p$, normalized physical time $\\tau=t/t_{\\mathrm{ref}}$, C-rate $C$, and ambient temperature $T_{\\mathrm{amb}}$. The fixed scale $t_{\\mathrm{ref}}=7200$~s is used for every condition. This avoids normalizing by each trajectory's final discharge time, which would reveal future information at inference.":
        "Figure~\\ref{fig:architecture} summarizes the computation. The inputs are $\\bar r=r/R_p$, $\\tau=t/t_{\\mathrm{ref}}$, C-rate $C$, and ambient temperature $T_{\\mathrm{amb}}$. We set $t_{\\mathrm{ref}}$ to 7200~s for every trajectory. A trajectory-specific end time is deliberately avoided because it is unknown until discharge finishes.",
    "Two fully connected subnetworks use four hidden layers of 96 neurons with hyperbolic-tangent activations. The concentration head receives $(\\bar r,\\tau,C,T_{\\mathrm{amb}})$; the lumped temperature head receives $(\\tau,C,T_{\\mathrm{amb}})$. Initial conditions are imposed by the output transformations":
        "The model has two multilayer perceptrons, each with four 96-unit tanh layers. Radius is supplied only to the concentration branch. The temperature branch uses time and operating-condition inputs, consistent with the lumped reference model. The following output forms satisfy both initial conditions exactly:",
    "with $\\Delta T_{\\mathrm{scale}}=50$~K. Stress is calculated from $\\widehat{c}_s$ using Eqs.~\\eqref{eq:A}--\\eqref{eq:st} on the evaluation grid.":
        "We use $\\Delta T_{\\mathrm{scale}}=50$~K for numerical scaling. Once $\\widehat{c}_s$ is evaluated on the radial grid, Eqs.~\\eqref{eq:A}--\\eqref{eq:st} produce both stress components.",
    "Automatic differentiation provides derivatives of neural outputs. In dimensionless coordinates, the diffusion residual is":
        "Automatic differentiation evaluates the time and radius derivatives. After substituting the normalized coordinates, the diffusion mismatch becomes",
    "Collocation points exclude the singular coordinate by using $\\bar r\\geq10^{-3}$; the exact center condition is penalized separately. The thermal residual is":
        "Interior samples begin at $\\bar r=10^{-3}$ so the spherical $2/\\bar r$ term is finite. A separate boundary term enforces the derivative at the true center. The energy-balance mismatch is",
    "where $\\widetilde{\\mathcal{R}}_T$ is divided by the characteristic thermal scale $\\rho c_p\\Delta T_{\\mathrm{scale}}/t_{\\mathrm{ref}}$. Scaling prevents units with large numerical magnitudes from dominating optimization.":
        "The tilde indicates division by $\\rho c_p\\Delta T_{\\mathrm{scale}}/t_{\\mathrm{ref}}$. This nondimensionalization keeps the thermal term from outweighing the other residuals solely because of its units.",
    "For labeled samples $\\mathcal{D}_L$, the proposed model uses normalized concentration and temperature-rise errors:":
        "At labeled points $\\mathcal{D}_L$, training compares normalized concentration and temperature rise:",
    "The total loss is $\\mathcal{L}=\\mathcal{L}_{\\mathrm{data}}+\\mathcal{L}_{\\mathrm{phys}}$. The data-only and prior-style baselines also learn radial and tangential stress labels; the proposed model does not require stress labels because stress follows from predicted concentration.":
        "The two terms are added with the weights shown in Fig.~\\ref{fig:architecture}. Both comparison models retain explicit radial- and tangential-stress outputs. The proposed network has no stress target in its supervised loss because mechanics is evaluated from $\\widehat c_s$.",
}
for old, new in replacements.items():
    replace_exact(old, new)

replace_between(
    r"\subsection{Reproducible Implementation Procedure}",
    r"\section{Experimental Methodology}",
    r"""\subsection{Reproducible implementation}
PyBaMM solves each C-rate and ambient-temperature pair as a separate simulation. The export contains physical time, negative-electrode interfacial current, heat generation, average temperature, and particle concentration on one time--radius grid. Training and evaluation use the following sequence:
\begin{enumerate}
    \item reserve complete operating conditions for training, validation, and testing;
    \item estimate normalization constants from training conditions and keep an unscaled copy of every variable used in a residual;
    \item sample labeled rows and physics collocation points independently within each training trajectory;
    \item optimize with Adam, retaining the checkpoint with the smallest validation objective and applying the stated patience rule; and
    \item reload that checkpoint, evaluate the untouched test conditions once, compute stress from concentration, and write the metrics.
\end{enumerate}
All variants receive identical condition partitions and identical labeled indices for a given seed. Physics points are resampled during optimization, whereas validation and test grids do not change. Each checkpoint is accompanied by the software version, parameter set, seed, label budget, chosen epoch, elapsed time, and trainable-parameter count.

The stress calculation needs an integral along particle radius. For each time and operating condition, predicted concentrations are ordered by radius and integrated with the cumulative trapezoidal rule. At the center, the limiting expression is evaluated directly. Replacing $r^3$ by an arbitrary small denominator would distort the center stress and can also change the reported peak.""",
)

replace_between(
    r"\subsection{Reference Data and Operating-Condition Split}",
    r"\begin{figure*}[!b]",
    r"""\subsection{Reference data and condition split}
The reference set contains constant-current SPMe discharges with lumped thermal coupling and Chen2020 parameters. A run stops at 2.5~V. We retain 100 time samples and 25 radial samples from each run. Training uses the Cartesian product of C-rates $\{0.5,1.0,1.5,2.0\}$ and ambient temperatures $\{10,25,35\}^{\circ}$C. Validation uses C-rates $\{0.75,1.75\}$ at $\{15,30\}^{\circ}$C. The two final test cases are 1.25C discharges at $20^{\circ}$C and $32.5^{\circ}$C. In total, the dataset has 18 trajectories and 45,000 radius--time records.

No trajectory is divided across subsets. Validation data determine early stopping and checkpoint choice; the test trajectories are read only after fitting has ended. Figure~\ref{fig:design} shows the sampled operating envelope and the matched ablation design.""",
)

replace_exact(
    r"\caption{Operating-condition split and ablation protocol. Every model uses the same conditions, label budgets, and seeds.}",
    r"\caption{Operating conditions and matched comparisons. Model variants differ in formulation, while partitions, label counts, and random seeds remain fixed.}",
)

replace_between(
    r"\subsection{Baselines and Training Protocol}",
    r"\begin{table}[t]",
    r"""\subsection{Models and optimization settings}
Three formulations are trained. The \textbf{data-only} network reproduces the earlier four-output design and fits concentration, temperature, radial stress, and tangential stress. The \textbf{prior-style PINN} keeps those outputs and adds diffusion, thermal, boundary, and mechanical-equilibrium penalties. The \textbf{proposed \modelname} removes both stress heads and generates stress from its concentration field.

For every formulation, each training trajectory contributes either 50, 150, or 300 labeled records. Seeds 11, 22, and 33 give 27 primary runs. Adam starts at $10^{-3}$; gradients are clipped at norm 10. Training ends at 2500 epochs or after 350 epochs without validation improvement. Each training condition supplies 900 interior and 120 boundary collocation points. Test evaluation always uses the saved validation-selected state.""",
)

replace_between(
    r"\subsection{Controlled Ablations}",
    r"\subsection{Evaluation Metrics}",
    r"""\subsection{Controlled comparisons}
The first comparison, data-only versus prior-style PINN, tests whether the added residuals help the existing four-output network. The second comparison changes two linked design choices: it replaces trainable stress outputs with the analytical operator and removes radius from the temperature branch. Results from this pair therefore measure the revised architecture as a whole. They cannot attribute every change to mechanics alone. Repeating both comparisons at 50, 150, and 300 labels reveals how their behavior changes as supervision becomes less sparse.

If computational time permits, three diagnostic runs can separate these effects: omit the thermal residual but keep temperature labels; apply the stress operator to concentration from the data-only network; and replace only the stress heads in the prior-style PINN while leaving its temperature input unchanged. These runs aid diagnosis but do not replace the predeclared three-model experiment.""",
)

replace_between(
    r"\subsection{Evaluation Metrics}",
    "\\begin{equation}\n\\mathcal{R}_{m}",
    r"""\subsection{Evaluation measures}
One score cannot describe a coupled field surrogate. We therefore report nine measures spanning prediction, physical checks, and runtime. For any reference field $y$ and prediction $\widehat y$ at $N$ locations,
\begin{align}
\operatorname{RMSE}(y)&=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\widehat y_i-y_i)^2},
\label{eq:rmse}\\
\operatorname{Rel}L_2(y)&=\frac{\|\widehat{\mathbf y}-\mathbf y\|_2}{\|\mathbf y\|_2}.
\label{eq:rell2}
\end{align}
The nine reported quantities are concentration relative $L_2$ error, surface-concentration MAE, temperature-rise RMSE, largest absolute temperature-rise error, tangential-stress RMSE, peak tensile-stress error, particle mass-balance RMSE, concentration--stress consistency RMSE, and median speedup relative to SPMe. MAPE is unsuitable for stress because the denominator approaches or crosses zero. Temperature scores use rise above ambient; division by absolute Kelvin temperature would make meaningful thermal deviations appear artificially small.

As a separate conservation check, the particle-average concentration must satisfy""",
)

replace_exact(
    "The consistency metric recomputes stress from predicted concentration and compares it with each model's reported stress. The proposed model has zero analytical inconsistency by construction, apart from floating-point and quadrature effects. For every primary metric, the paper reports mean $\\pm$ standard deviation across seeds and the worst held-out condition. Improvement over the prior formulation is",
    "For the coupling check, stress is recomputed from each model's concentration and compared with that model's stress output. In the proposed architecture these two quantities are the same construction, leaving only floating-point and quadrature error. Results include the seed mean, sample standard deviation, and the poorer of the two held-out conditions. Relative change from the prior-style model is",
)
replace_exact(
    "where positive values indicate lower proposed-model error.",
    "A positive $\\Delta_m$ means that the revised model has the smaller error.",
)

replace_between(
    r"\subsubsection{Inference Latency and Solver Speedup}",
    r"\begin{figure}[!b]",
    r"""\subsubsection{Inference latency and solver speedup}
Runtime is measured with the same held-out query batch, numerical precision, and computer for both methods. Let $\widetilde t_{\mathrm{PINN}}$ and $\widetilde t_{\mathrm{SPMe}}$ be the median times across repeated blocks. We define
\begin{equation}
S_{\mathrm{lat}}=\frac{\widetilde t_{\mathrm{SPMe}}}{\widetilde t_{\mathrm{PINN}}}.
\label{eq:speedup}
\end{equation}
The timing loop performs 100 warm-up calls and synchronizes pending operations before recording the median and 95th percentile. A ratio above one favors PINN inference. Training duration is listed separately because it is a one-time cost and is not part of per-query latency.""",
)

replace_exact(
    r"\caption{Latency protocol using identical held-out queries and hardware, with warm-up and synchronization before PINN--SPMe comparison.}",
    r"\caption{Timing procedure for the PINN and SPMe reference solver on one held-out query batch.}",
)

replace_between(
    r"\subsection{Statistical Summary and Acceptance Criteria}",
    r"\section{Results and Discussion}",
    r"""\subsection{Aggregation and decision rule}
Metrics are first computed for each test condition and seed. For a given model and label budget, the main summary averages the two conditions within each seed and then reports the mean and sample standard deviation over the three seeds. The worse test condition is retained as a separate column; an average can otherwise hide a temperature-specific failure. With only three seeds, the standard deviation describes run-to-run variation and is not treated as a significance test.

Support for the hypothesis requires more than a zero coupling residual. The revised model must remove the designed concentration--stress mismatch, avoid a material increase in stress error, and keep concentration and temperature accuracy comparable across label budgets. Timing claims are valid only under the fixed procedure in Fig.~\ref{fig:latency_protocol}. Parameter count and training duration are supporting resource measures.""",
)

replace_between(
    r"\subsection{Result-Reporting Contract}",
    r"\begin{table*}[t]",
    r"""\subsection{Result-reporting rule}
Table~\ref{tab:results} reads values produced by the notebook summaries \texttt{paper\_metrics\_summary.csv} and \texttt{variation\_gap\_vs\_previous.csv}. Placeholder text is retained in this draft because the complete three-seed experiment has not yet been inserted. A submission version must run with \texttt{quick\_mode=False} and replace every macro at the start of the source with the measured mean and standard deviation. No numerical result is inferred from an incomplete run.""",
)

replace_between(
    r"\subsection{Required Plots and Traceability}",
    r"\section{Limitations and Threats to Validity}",
    r"""\subsection{Plots and traceability}
The notebook exports plots of error against label count, surface concentration, temperature rise, and both stress components. A separate panel shows the mass-balance residual. Each plot names the held-out condition and seed aggregation. Scalar scores provide a compact comparison, while trajectories reveal timing offsets, boundary errors, and short stress peaks that an average may hide.

All summaries are built from the same condition-level results file. Before aggregation, the script checks for the 27 expected runs, verifies that condition groups do not overlap, and confirms three seeds for each reported mean. An unsuccessful run remains visible as incomplete rather than being dropped from the denominator.

\subsection{Questions answered by the experiment}
The experiment tests whether residual training helps when labels are scarce and whether the analytical mechanics layer improves agreement between concentration and stress without sacrificing the two learned fields. The evidence must include all seeds, both test conditions, peak and field stress errors, and the independent conservation calculation. The test points lie inside the sampled C-rate and ambient-temperature range. They say nothing about charging, arbitrary drive cycles, different chemistries, or extreme temperatures.

\subsection{Expected physical behavior}
During discharge, reaction at the particle surface can create a radial concentration gradient and hence diffusion-induced stress. A spatially uniform profile gives zero stress under Eqs.~\eqref{eq:sr} and \eqref{eq:st}; faster discharge is expected to increase transient nonuniformity. Equation~\eqref{eq:thermal} links temperature rise to generated heat and convective removal. Unlike parameter-inference PINNs \cite{hassanaly2024part1,hassanaly2024part2}, aging-oriented models \cite{wei2026etm}, and life-cycle temperature predictors \cite{wang2023temperature}, this experiment evaluates within-discharge agreement between electrochemical and mechanical fields. These physical expectations are checks on the measured curves, not substitutes for them.

\section{Limitations and Threats to Validity}""",
)

replace_between(
    r"\section{Limitations and Threats to Validity}",
    r"\balance",
    r"""\section{Limitations and Threats to Validity}
Every target in this study is simulated. Agreement with PyBaMM shows that the surrogate reproduces the selected SPMe configuration; it does not constitute experimental validation. The model represents one negative-electrode particle, one cell-average temperature, fixed material properties, and constant-current discharge. It omits through-thickness thermal gradients, spatially varying electrode stress, plastic deformation, binder behavior, fracture initiation, and two-way stress--diffusion feedback.

The elastic constants are literature values rather than measurements from the modeled LG M50 cell. The resulting stress should be interpreted as a model-derived indicator. Chen2020 parameters were assembled from multiscale experiments, but a parameter set remains an input to the mathematical model rather than proof of every internal property of a particular cell \cite{chen2020parameterization}.

Heat-generation histories also come from the SPMe reference. The neural network interpolates within the selected constant-current conditions and has not been tested with an arbitrary current waveform. PINN optimization adds another limitation: data and residual terms can produce gradients of very different sizes. Repeated seeds and independent residual calculations expose some unstable fits but cannot remove error in the governing assumptions.

The study makes no aging claim. PyBaMM can represent SEI growth, plating, active-material loss, and cracking \cite{sulzer2021pybamm}, but a credible degradation experiment would also need cycling schedules, health or capacity targets, degradation parameters, and held-out cells or profiles. Stress may later act as an input to such a model; it is not interchangeable with degradation.

\subsection{Sensitivity and numerical uncertainty}
Equations~\eqref{eq:sr} and \eqref{eq:st} scale stress by $E\Omega/(1-\nu)$. Errors in $E$ or $\Omega$ can therefore dominate the mechanical result even if concentration is accurate. The planned sensitivity calculation passes each stored concentration field through the operator after changing $E$ and $\Omega$ separately by $\pm20\%$. It records the change in peak stress without retraining. Because $D_s$ is fixed during fitting, conclusions apply to the selected Chen2020 diffusivity.

The radial integral is repeated with 25, 50, and 100 grid points. Peak stress and mass-balance error should approach stable values as the grid is refined; otherwise the reported difference may be a quadrature artifact.

\subsection{Failure analysis}
The saved histories allow failures to be assigned to specific stages. High concentration error with a small PDE residual suggests a boundary, scaling, or initial-condition problem. Low label error with a large off-grid residual points to poor physical generalization. If concentration is accurate but stress is not, the elastic constants or radial integration should be checked. A nearly constant temperature bias suggests heat capacity or heat generation, whereas an error that changes with ambient temperature points toward the cooling term.

Non-finite losses, very early checkpoints, and runs still improving at epoch 2500 are reported with their seeds rather than replaced. The held-out cases measure interpolation only. Robustness would require additional tests with noise, parameter shifts, and current profiles. A small residual confirms agreement with the equations imposed during training; it cannot establish that those equations include every relevant process.""",
)

replace_between(
    r"\section{Conclusion}",
    r"\section*{Data and Code Availability}",
    r"""\section{Conclusion}
This work revises a battery PINN so that its mechanical field is tied directly to its electrochemical field. The trainable outputs are particle concentration and lumped temperature. Radial and tangential stresses follow from the concentration integral, while diffusion and energy-balance residuals regularize the learned fields. The evaluation keeps complete operating conditions separate and repeats matched model variants across label budgets and seeds.

The final claim depends on the full experiment. The redesign is useful only if it removes concentration--stress disagreement without materially worsening stress, concentration, or temperature error. If consistency improves but prediction does not, the next step is to improve concentration learning or calibrate the mechanical constants rather than claim success from the architectural identity alone. Experimental cells, dynamic-current profiles, and cycling data are needed before extending the conclusions to degradation or battery health.""",
)

replace_exact(
    "The accompanying Colab notebook generates PyBaMM data, trains all baselines, and exports metrics, checkpoints, and the label-budget ablation. Data and weights should be archived with the software versions and seeds in Table~\\ref{tab:config}.",
    "The project notebook generates the PyBaMM trajectories, trains the three formulations, and writes checkpoints, per-condition metrics, and label-budget comparisons. An archived release should include the data, weights, software versions, and random seeds listed in Table~\\ref{tab:config}.",
)

path.write_text(text, encoding="utf-8")
print(path)
