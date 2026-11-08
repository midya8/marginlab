# Exact finite input boxes, without a clinical claim

MarginLab v0.3 adds an original educational implementation of established affine support functions. It changes no fit, score, split, source byte, training-objective certificate or model selection. The original clinical/generalization exclusions still apply.

## The declared function

The saved binary64 training rows Z, signed integer labels y, coefficients alpha and bias b define the exact rational standardized affine function

```
w_j = sum_i exact(alpha_i) y_i exact(Z_ij)
f(x) = exact(b) + sum_j w_j (x_j - exact(mean_j)) / exact(scale_j)
a_j = w_j / exact(scale_j)
c = exact(b) - sum_j a_j exact(mean_j)
f(x) = c + a*x
```

`exact` means the precise fraction of the represented binary64 value, not its shortest printed decimal and not a rounded `w`. In particular, `Fraction(0.1)` differs from `Fraction('0.1')`. The source rows Z were computed with floating subtraction/division during the original fit. This work freezes those represented rows and scalers; it does not reconstruct exact real-number standardization. Browser source matching recomputes every represented Z from the checked public raw source and scalers, then independently derives the same a and c using BigInt. The reference test checks all four affine declarations against native values. The existing scalar floating inference is shown separately, not certified by the rational interval.

## Finite original-unit intervals

A request declares a rational center x0 and lower/upper l,h, with l_j <= x0_j <= h_j. For every coordinate,

```
minimum_j = min(a_j*l_j, a_j*h_j)
maximum_j = max(a_j*l_j, a_j*h_j)
minimum = c + sum_j minimum_j
maximum = c + sum_j maximum_j
```

If a_j >= 0 the minimizing corner uses l_j and the maximizing corner h_j. Reverse this for a_j < 0. For a_j=0 any in-box coordinate attains both. The exported proof contains the full declared function/set, center score, both corner vectors, both scores and tie-aware label claims. Independent readers check membership, each coordinate support equality, scores and label claims. This proves attainment, not merely an outer interval.

Decision rule: +1 when f>=0, otherwise -1. Thus -1 is reachable iff minimum<0; +1 iff maximum>=0. The whole box has one label iff minimum>=0 or maximum<0. Equality cases are intentionally asymmetric. This is label stability relative to the frozen declared function, not correctness against any historical or medical target.

## Synthetic presets, not measured tolerances

The inspector starts from eight unchanged historical test feature vectors, no original identifiers, with all 30 features in original units. A plain decimal spread e gives exact halfwidth e*exact(training_scale_j) and endpoints exact(center_j) +/- halfwidth_j. `0.1` in this control is exactly 1/10. There is no implicit clipping, covariance model, admissible-feature manifold, measurement-accuracy claim, sensitivity/specificity estimate or causal intervention. Boxes may contain negative or physically implausible coordinates. Editing intervals invalidates the previous result immediately; out-of-box centers and malformed inputs refuse without a fallback result.

UI approximate decimals/plot positions are not the proof. Large exact numerator/denominator pairs are approximated from leading digits and decimal lengths rather than infinity/infinity; native/browser exact fractions and comparisons are unchanged. The plot uses a zero-including common scale and responsive labels. Raw endpoint fractions remain in the full witness; the shorter attaining-corner display is approximate.

## Reproduce offline

```
python -m marginlab.box_export
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m marginlab box --model-id linear-1 --input examples/box.json --output /tmp/box-proof.json
python -m marginlab verify-box --input /tmp/box-proof.json
```

`examples/box.json` contains only `center`, `lower`, `upper` arrays of canonical n/d strings. The command derives the affine function from the saved report, or `--report` for a supplied report. `verify-box` checks internal mathematics only; it cannot authenticate a model, input, author or artifact. Changing a declared affine function can produce another valid proof. Browser source consistency is a separate local check, not cryptographic authentication. RBF requests fail explicitly. JSON duplicate keys, unknown fields, booleans, numeric rather than rational input, noncanonical signs/denominators, empty/mismatched vectors and complexity excess refuse. Up to32 features,512-bit vector entries,4096-bit affine bias, bounded rational components/readers prevent unbounded witness parsing; extreme arithmetic outside reader bounds also refuses, not rounds.

Package reproduction imports every module outside checkout and executes the original commands plus box, verify-box and the box exporter. Four readable artifact chunks below600KB preserve128 full responses:4linear fits x8fixed queries x4spreads. Browser downloads need no API, network endpoint or paid service.

## Independent checks

-120 tiny1-5D exact cases, all corners exhaustively enumerated independently, including mixed signs, zeros and asymmetric intervals.
-128 complete saved-source reference responses. All256 extremal scores are independently recomputed from the exact dual sum `b + sum_i alpha_i*y_i*sum_j Z_ij*((endpoint_j-mean_j)/scale_j)`, not the affine support implementation.
-Independent Python Fraction/native and JavaScript BigInt browser algorithms, full proof equality, four source-declaration matches.
-Tie cases, zero weights/nonunique corners, point boxes, negative spreads, invalid fractions, field/type/complexity refusal and duplicate JSON.
-Nonmutation of saved fits; valid changed-function control distinguishes internal proof validity from source identity. Actual browser source tampering clears all results and refuses export.
-Actual Chrome128model/query/spread combinations, finite displayed metrics, keyboard selection, edited-state invalidation, invalid intervals/spread, denied/successful clipboard, actual Blob JSON and320/390/1280layout.

Existing60rational active-face optima, pair replay, eight fresh fits, frozen-holdout poisoning,4,552native/browser score checks, exact weak-duality proofs and all controls remain. None of these tests establishes clinical reliability, domain-admissible input sets, RBF robustness or floating-point implementation safety.

## Sources and attribution

1. Pantelis Sopasakis, [Support functions](https://am-press.github.io/posts/maths/support-functions/),2022-07-12. Definition, dual-norm relationship and coordinate-separable support motivation. Our finite-box proof uses the definition directly, not the later polyhedral halfspace formulas.
2. Giovanni Chierchia, [Box constraints](https://perso.esiee.fr/~chierchg/optimization/content/05/box.html). Nonempty finite intervals and bounds. Projection is not our optimizer; no copied code.
3. Goodfellow, Shlens and Szegedy, [Explaining and Harnessing Adversarial Examples](https://research.google/pubs/explaining-and-harnessing-adversarial-examples/),ICLR2015. Linear dependence as adversarial-example motivation. No neural-network, training-improvement or empirical robustness claim transferred.
4. Fawzi, Fawzi and Frossard, [Analysis of classifiers' robustness to adversarial perturbations](https://ar5iv.labs.arxiv.org/html/1502.02590). Distinguishes adversarial/random perturbations and notes general nonlinear search need not attain an optimum. Their distributional risk/generalization analysis and Euclidean experiments are not our finite-box result.
5. Python documentation, [fractions](https://docs.python.org/3/library/fractions.html). Exact conversion of represented floats, canonical rational arithmetic. We do not depend on newer formatting APIs; runtime remains Python3.10+.
6. Python tutorial, [Floating-point issues and limitations](https://docs.python.org/3/tutorial/floatingpoint.html). Decimal/binary distinction and operation rounding. Exact affine arithmetic is explicitly separate from existing machine inference.
7. Maurya, Barik and Honorio, [On Exact Solutions of the Inner Optimization Problem of Adversarial Robustness](https://arxiv.org/html/2208.09449v3). Related exact inner optimization and norm constraints. No new algorithm, adversarial-training, Rademacher/generalization bound or experiment claim is borrowed.

Established mathematics, original code and tests, MIT. Existing UCI data and attribution remain CC BY4.0. Attribution is not endorsement.
