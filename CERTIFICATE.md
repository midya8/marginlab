# Exact declared linear weak-duality bounds

A numerical fit and a feasible exact dual witness are different objects. This extension leaves all original fits, data, split, scores, browser references and score figure unchanged. Only the four fixed linear models are supported; the RBF exponential kernel is explicitly rejected.

## Which problem is proved

Each finite saved standardized training coordinate, C and bias is read as its exact binary64 rational value. This is the represented standardized training problem, not exact real-number standardization of physical measurements. The primal is

P(w,b,xi) = (w dot w)/2 + C SUM_i xi_i,

with xi_i >= 0 and y_i(w dot x_i+b)+xi_i >= 1. The dual is

D(a) = SUM_i a_i - ||SUM_i a_i y_i x_i||^2/2,

with 0 <= a_i <= C and SUM_i a_i y_i = 0. For any feasible primal and dual, weak duality gives D <= optimum <= P. Thus P-D is an exact suboptimality upper bound for that particular feasible primal. A nonzero bound does not assert attainment of the exact optimum. It is not a generalization, medical-reliability or calibration bound.

## Equality repair, not tolerance laundering

The original saved alpha has a small nonzero exact signed mass. For C=.01,.1,1,10 the respective residuals are 7/2^60, 7/2^58, 105/2^58 and -287/2^54. Those original values cannot simply be called exact feasible dual points.

1. Convert binary64 inputs to exact Fractions, not intended decimal strings.
2. Clip each original coefficient to the exact [0,C] interval.
3. Compute the exact excess signed mass. Visit indices in source order and reduce available coefficients in the excess class until that mass is exactly zero. There is always enough excess-class mass to reduce. Reductions cannot break either box bound.
4. Use the repaired coefficients for the dual objective. Record the original equality residual, original box residual, clipped equality and total L1 coefficient movement.
5. Use the exact linear combination of the ORIGINAL coefficients as the primal w, the exact saved bias and xi_i=max(0,1-y_i(w dot x_i+b)). Any w is allowed in the primal, so the original dual equality is irrelevant to primal feasibility.

The two witnesses deliberately need not use the same coefficients. No original prediction or model parameter is changed. Approximate exact gaps are 7.97e-8, 1.42e-6, 2.90e-5 and 3.23e-4. Full objective ratios, not the approximate decimal readouts, are the actual bounds.

## Independent readers

`certificate.py` produces the self-contained problem, primal/dual witnesses and repair accounting. `verify_certificate.py` separately derives every margin, box, equality and objective using Fraction and does not import the producer, optimizer or a Gram matrix. The browser reader separately implements normalized BigInt rationals and exact IEEE-754 decoding with DataView. Neither independent reader runs a fit or trusts stored objectives. It recomputes them.

Sixty additional active-face exact optima for tiny integer-feature/dyadic-C problems must lie within produced intervals. All four full 341x30 proofs pass both independent readers. Tests cover both signs of equality repair, out-of-box clipping, zero budget, zero curvature, infeasible margins/dual points, changed objective/repair/schema fields, noncanonical or oversized fractions, unsupported RBF, strict JSON duplicate keys, native/browser agreement and installed CLI reproduction. Original optimization and holdout tests remain in place.

The readers impose input size and canonical ratio limits. They check a mathematical witness; they do not authenticate its author, source, or intended use. Python CLI duplicate keys and nonfinite JSON numbers are rejected.

## Validity is not source authentication

A deliberately changed inactive training row can still satisfy all exact constraints without changing either objective. That proof remains mathematically valid for the CHANGED declared problem. It does not prove it describes UCI. This is an explicit passing control, not an unexplained failure to reject tampering.

The browser separately compares every enclosed standardized input, label, C, original alpha and saved bias to the checked original model. That source-match check is separate from mathematical proof validity. A substituted page and substituted proof are not authenticated by these comparisons. There is no signing, dataset-origin authentication or clinical validation claim.

## Reproduce

After `python -m pip install -e .`:

```sh
python -m marginlab certificate --model-id linear-1 --output proof.json
python -m marginlab verify-certificate --input proof.json
python examples/certificate_export.py
python -m unittest discover -s tests -v
node --test tests/*.cjs
```

The copy and download controls export the full self-contained proof, not only the displayed summary. The short decimal display is approximate and must not be treated as an outward-rounded interval. The full canonical rational strings are exact. The original inspector continues to label its original full-data gaps numerical.

## Sources

- Platt's primary SMO report: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf . Dual objective and box/equality constraints, not a claim to implement every original pair-selection heuristic.
- Berkeley soft-margin lecture: https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/lectures/lec6.pdf . Primal sum-hinge, constraints and dual convention.
- CMU duality lecture: https://www.cs.cmu.edu/~atalwalk/teaching/winter17/cs260/lectures/lec13.pdf . Feasible weak-duality bounds.
- Python Fraction documentation: https://docs.python.org/3/library/fractions.html . Exact float conversion and normalized rationals; `from_float(0.3)` is not 3/10.
- Python floating-point tutorial: https://docs.python.org/3/tutorial/floatingpoint.html . Exact binary64 ratios, not intended decimal values.
- Original public data, license, exact hashes and clinical limitations remain documented in [METHOD.md](METHOD.md).
