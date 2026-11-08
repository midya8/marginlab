# Soft-margin learning with explicit residuals

This is an original implementation of established SVM dual optimization, not a new algorithm or a clinical product. [Live inspector](https://midya8.github.io/marginlab/) · [Source](https://github.com/midya8/marginlab)

## Objective and conventions

Labels y are -1 and +1. We use `f(x)=sum_i alpha_i y_i K(x_i,x) + b`. Platt's original report uses a threshold with a minus sign; our bias convention is deliberately the opposite.

Primal `P = 0.5 ||w||² + C SUM_i max(0,1-y_i f(x_i))`.

Dual `D = SUM_i alpha_i - 0.5 (alpha*y)^T K (alpha*y)`, subject to `0 <= alpha_i <= C` and `y^T alpha=0`.

C multiplies the **sum**, not the mean, of hinge losses. Our C values are not interchangeable with a mean-loss regularization constant. Linear K is the inner product. RBF `K(a,b)=exp(-gamma ||a-b||²)`, with gamma fixed at 1/30. Both are positive semidefinite; strict positive definiteness is not required. NumPy supplies array operations, not an SVM solver.

The bias is unregularized. For exact feasible dual variables, weak duality makes the dual a lower bound and the primal a corresponding upper value. The gap bounds objective suboptimality, not prediction error, calibration or medical reliability. Floating arithmetic can leave a small equality residual, so our displayed gap is a **numerical gap with separate feasibility residuals**, not an exact certificate.

## Deterministic feasible-pair rule

Start alpha=0. Let `raw=K(alpha*y)` and `h=y-raw`.

- I_up: positive y with alpha<C, or negative y with alpha>0.
- I_low: positive y with alpha>0, or negative y with alpha<C.
- Choose i with maximum h in I_up, j with minimum h in I_low. Ties use source training order.
- Direction: `delta alpha_i=t*y_i`, `delta alpha_j=-t*y_j`. This preserves the linear equality in exact arithmetic.
- Curvature `eta=K_ii+K_jj-2K_ij` is nonnegative for supported kernels. The dual directional derivative is `v=h_i-h_j`.
- The upper feasible t is the smaller of the two directional box slacks. For eta>0, maximize the line quadratic with `t=min(limit,v/eta)`. For eta=0 and positive v, choose the endpoint. Never divide by zero or silently skip coincident opposite-label examples.
- The predicted dual gain is `t*v-eta*t²/2`; nonpositive/no finite movement returns `no_progress`, not a convergence label. Tiny negative curvature beyond tolerance fails explicitly.

Raw scores update by the actual two represented alpha changes. When a bound is reached, the limiting alpha is set to its exact box endpoint and its partner is derived from the preserved equality. Scores refresh from the full Gram product every 100 updates and before stopping. All pair indices, two alpha changes, curvature and predicted gains are retained. A numerical replay independently recomputes every pair's curvature/gain and checks box/equality residuals along the path, final parameters and telescoping total dual gain. This does not authenticate model origin or represent exact arithmetic.

This is an SMO-style two-variable method with a deterministic maximum-violation rule, **not** a reproduction of Platt's full non-bound/error-cache/randomized fallback heuristics or a claim about their runtime advantage. Dense Gram storage is quadratic in training count; no large-scale solver claim is made.

## Bias, KKT and stopping

Set `b=(max(h[I_up])+min(h[I_low]))/2`. At an optimum these delimit a feasible bias interval; during fitting the interval can be reversed and represents violation, not certified feasible bounds.

For the represented final margins m=y*f, the KKT requirements are:

- alpha=0 implies m>=1;
- 0<alpha<C implies m=1;
- alpha=C implies m<=1.

These are implications, not equivalences. A margin-boundary example can have alpha=0, and duplicated examples can split coefficients differently. Bound membership uses the actual represented alpha, without hiding a second undocumented support threshold.

A `kkt_and_gap_tolerance` status requires both `max(h[I_up])-min(h[I_low]) <= 1e-5` and `P-D <= 1e-5*(1+abs(P))`, recomputed from the full Gram matrix. Box and equality residuals remain visible separately. A budget of zero is valid; budget exhaustion and no progress remain distinct. Every real fit uses 50,000 maximum accepted updates. All eight reach the numerical criteria on this split; that is not a universal convergence or exact-optimum certificate.

## Frozen data protocol

Original UCI Wisconsin Diagnostic data: 569 rows, 30 numeric features, 357 B and 212 M labels. Row ID is excluded from all model features and browser data. M maps to +1 and B to -1. There are no missing values, duplicate identifiers or exact duplicate feature vectors in the bundled original file. Those checks do not establish independent patients, site separation or clinical validity.

The source page explicitly grants CC BY 4.0. Original `wdbc.data` SHA256 `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`; `wdbc.names` SHA256 `840e04e3f20f8a5b326892f3b9cbc01c4cd6f7e6c597630b701ef6c0ac79f5ef`. The data bytes are bundled unchanged. The names file is retained losslessly as a JSON string in `wdbc.names.json`; decoding gives exactly the original bytes, without adding a trailing newline. A generated, hash-checked split manifest records seed 20261108, class-stratified 60/20/20 floor cut points and sorted original source-row indices: 341 training, 113 validation, 115 test. This is not an official UCI test set. Class labels are used to stratify once, not to fit scaling or model selection on test.

Means and population standard deviations use training rows only. Linear C in {0.01,0.1,1,10}; RBF C in {0.1,1,10,100}, gamma=1/30. These options, budget, tolerance and split are fixed before test scoring. Every model trains on the same training rows. Select maximum validation balanced accuracy, then minimum validation mean hinge, then fixed configuration order. No refit on validation and no choice based on test ranking. Poisoning frozen test features and labels changes neither models, pair histories, scale nor selection.

A full-30D standardized nearest-neighbor control and training-majority control use the same data and test stage. Nearest-neighbor is not restricted to two displayed features. Hinge scores for +/-1 baseline predictions are reported as such, not probabilistic losses.

## The real result

Validation selects linear C=1. It gets 112/115 test labels correct, compared with 111/115 for 1NN and 72/115 for training-majority. RBF C=10 and C=100 each get 113/115, but are not selected by test. A one-case difference on this small split is not evidence of general superiority. All eight candidates, both controls and train/validation/test summaries stay visible. The selected model's numerical gap is about 0.0000290 and its maximum KKT residual about 0.00000497.

## Controls and independent checks

1. Four separable 1D examples already have 100% training accuracy after a single pair step, yet a positive objective gap and `budget_exhausted` status. Correct labels do not mean optimization is solved.
2. Coincident opposite labels have eta=0. The linear endpoint update reaches alpha=[0.7,0.7], P=D=1.4, gap=0. Skipping zero curvature would miss the optimum.
3. Duplicated same-label points admit different support allocations with the same optimal decision function. Dual alpha is not unique causal data importance.
4. Sixty independently generated tiny integer/rational problems are checked by an independent `Fraction` active-face KKT solver. It enumerates box-bound/free faces and solves exact augmented linear systems, using no pair steps or NumPy solver. This is an exact oracle for those supplied tiny problems, not an exact certificate for the full floating real-data fits.
5. Independent scalar kernel calculations, PSD checks, analytic two-point solutions, label inversion, orthogonal rotations, translations, zero budget, invalid inputs and strict CLI errors cover numerical mechanics.
6. Full pair replay checks all eight saved fits and eight fresh fits. Saved-model/browser comparisons cover all eight models x 569 examples, 4,552 scores. The actual Chrome page checks 24 model/example states, keyboard input, editable features, invalid export blocking, clipboard failure/success, a real JSON blob and 320/390/1280 layouts.

## What the browser shows

All 30 original-unit features are editable. Scores and signed terms are recomputed with separate scalar JavaScript. For linear models, terms w_j*z_j sum with the bias to the score; for RBF models, displayed signed training-example terms are alpha_i*y_i*K(x_i,x). These are algebraic decompositions, not unique causal attributions. Full terms are exported.

The colored surface is a **decision slice** in the first two standardized features, with the other 28 fixed to zero. Training points overlay only their first two coordinates. Their full-input decisions need not equal the decision-slice color at that location. This is not a projection of the 30D boundary or a model trained on two features.

A large score at an extreme edited input is not calibrated certainty. No specimen plausibility, patient-level diagnosis, deployment readiness, regulatory approval, medical utility or population guarantee is claimed. Synthetic controls are descriptive mathematical examples, not causal component ablations of the historical task.

## Sources

- Platt, J. (1998), *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*, Microsoft Research TR-98-14. Read primal/dual, KKT, line bounds, positive and zero curvature, threshold and heuristic sections. Established algorithm context; our max-violation selection differs. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
- Platt, J. (1998), *Using Analytic QP and Sparseness to Speed Training of Support Vector Machines*, NeurIPS. Primary paper's two-variable subproblem. Its historical speed claims are not reproduced here. https://proceedings.neurips.cc/paper_files/paper/1998/file/7e1d842d0f7ee600116ffc6b2d87d83f-Paper.pdf
- Jordan, M., Berkeley CS281B/Stat241B, *Soft Margin SVM*. Primal, box-constrained dual, unregularized bias and KKT implications. https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/lectures/lec6.pdf
- Talwalkar, A., CMU CS260 (2017), lecture 13. Convex SVM, weak/strong duality and necessary/sufficient KKT context. https://www.cs.cmu.edu/~atalwalk/teaching/winter17/cs260/lectures/lec13.pdf
- Dillhoff, A., *Sequential Minimal Optimization* (2022, updated February 8, 2026). Secondary elementwise/line-constraint cross-check. Some displayed clipping/derivation statements have typos; the original report and independent oracle, not those expressions, govern the implementation. https://ajdillhoff.github.io/notes/sequential_minimal_optimization/
- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, DOI 10.24432/C5DW2B, CC BY 4.0. Dataset, target definition, features, citation and explicit license footer checked on the original page. https://archive.ics.uci.edu/dataset/17/breast%C2%B1cancer%C2%B1wisconsin%C2%B1diagnostic
- Original UCI download, linked by that page: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip

MIT code; CC BY 4.0 original data. Attribution does not imply source-author endorsement. Established methods, original implementation; no new optimization or diagnostic method.

## Bounded artifact storage

The complete report is assembled from `marginlab/artifacts/report.json`, eight readable per-candidate JSON files and three bounded linear-10 pair chunks. `saved()` restores all original values and pair order. Each stored text file is below 600 KB to avoid large-file browser-editor truncation. Fresh `fit --output` still writes one complete ordinary JSON report. The source names text is stored as a JSON string so its exact original bytes, including its missing final newline, survive text-editor normalization. Both original source hashes are verified after decoding. This is storage only, not a refit, rounding or change of protocol.

## Cross-platform reproduction

Floating-point reductions can change near-tied pair choices: the published artifact has 275 updates for linear C=0.01, while a Python 3.10 CI run took 267. Reproduction requires the same protocol, selected configuration, stop status and all train/validation/test confusion counts, overlapping numerical primal/dual objective intervals and full pair replay. It does not require identical update counts or dual coefficients, which need not be unique. The full-data intervals are still numerical, with a comparison slack of 5e-7 times (1 + maximum absolute primal value), not exact bounds. The holdout-poison test compares two fresh fits in the same runtime and requires exactly identical fitted models, pair histories and validation metrics. Independent metric recomputation allows 1e-12 times (1 + absolute hinge) for the mean hinge reduction; confusion counts remain exact. No saved model, score, stopping tolerance or data split is changed by these test corrections.


## Exact represented linear extension (v0.2)

The original numerical gaps above remain numerical. A separate extension constructs exact rational primal and repaired-dual feasible witnesses for the four saved linear models. It does not change any fitted coefficient, prediction, split, pair history or old objective. [CERTIFICATE.md](CERTIFICATE.md) derives the repair and weak-duality interval, identifies the declared binary64 problem, and separates mathematical validity from source matching. The independent Fraction and BigInt readers do not call the optimizer. RBF and generalization bounds are not supported.
