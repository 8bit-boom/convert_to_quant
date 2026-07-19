# Learned Rounding Auto-Tune Review

## Purpose

This note records the current learned-rounding auto-tune design, its limits, and a safer direction for making tuning genuinely responsive to layer behavior. It also defines the narrow boundary where comfy-kitchen may assist INT8 quantization without changing convert-to-quant's runtime contract.

This is an engineering review, not authorization to replace the current implementation. Auto-tune should remain optional until a redesigned controller demonstrates quality parity with strong manual settings across representative models and quantization formats.

## Compatibility Invariants

Future tuning and backend work must preserve all of the following:

- `.comfy_quant` tensors and `_quantization_metadata` remain part of the saved model contract.
- The legacy `int8_tensorwise` format name remains supported. The name does not imply that the scale must be scalar.
- Runtime scalar versus per-channel behavior continues to be determined from the scale tensor shape.
- Legacy scalar scaling remains loadable even though current conversion primarily uses row or per-channel scaling.
- Simple and learned quantization routes remain available through the CLI and exposed quantization functions.
- Learned ConvRot retains its exact rotated floating-point weight, rotated calibration activations, unquantized target, rounding optimization, and bias correction in the rotated basis.
- Explicit-scale requantization used by learned rounding remains under convert-to-quant's control.
- Blockwise INT8, metadata fields, tensor names, scale shapes, routing, and serialization are not changed as a side effect of backend integration.

These invariants make a broad replacement of the local INT8 conversion path inappropriate. Backend calls must be evaluated operation by operation.

## Current Auto-Tune Behavior

### Corrective implementation status

The first corrective pass removes the most direct violations identified by this review:

- `num_iter` is treated only as the maximum layer budget.
- Restart-based learning-rate probes are disabled.
- Each layer uses one continuous selected optimizer attempt unless instability requires bounded recovery.
- The user-selected scheduler is preserved; auto-tune only observes convergence and instability.
- The legacy `adaptive` scheduler is never selected implicitly.
- Prodigy always starts at `lr=1.0`, including manual and automatic routes.
- AdaRound temperature and NVFP4 scale-refinement timing use an internal 2,000-step policy horizon rather than the maximum budget.
- Strict best-state capture is separated from noise-aware convergence significance.
- Reports use schema version 2 and distinguish configured from effective starting learning rate.

The remaining sections retain the original audit findings because they explain why the correction was necessary and what must be measured before introducing a later intelligent policy.

The auto-tune entry point is `BaseLearnedConverter._run_auto_tuned_optimizer` in `convert_to_quant/converters/base_converter.py`. Loss-trend control is implemented by `AdaptiveConvergenceController` in `convert_to_quant/converters/convergence.py`.

For each eligible learned optimization, the current implementation:

1. Infers a two-dimensional weight shape and, where possible, a factorization rank.
2. Computes a deterministic observation window from rank and matrix aspect ratio. The window is clamped between 16 and 128 iterations.
3. Uses `num_iter` as a maximum per-layer budget without deriving policy timing from it.
4. Runs one continuous selected attempt at the optimizer-family starting learning rate.
5. Preserves the explicitly selected plateau, exponential, or adaptive scheduler.
6. Observes normalized loss independently for convergence and instability.
7. Allows one bounded retry only when the controller reports instability and enough budget remains.
8. Restores the converter's original tuning arguments after the attempt and can write per-layer diagnostics through `--auto-tune-report`.

Within an attempt, the controller:

- Normalizes loss against the first observed loss of that attempt.
- Estimates recent noise with the median absolute deviation of normalized step deltas.
- Treats an improvement as meaningful only when it exceeds `max(1e-6, 3 * noise)`.
- Marks a window as plateaued when its relative gain is no greater than the same style of noise floor.
- Applies a `0.8` learning-rate factor for a quiet plateau and `0.5` for a noisy plateau.
- Stops for negligible normalized loss, a sufficiently stable plateau after at least one reduction, a non-finite loss, or sustained loss above four times the initial value.
- Recommends retry only for non-finite loss or sustained regression.

This corrected design is deterministic and shape-aware, but its decisions remain primarily fixed heuristics over loss history. It should not yet be described as intelligent parameter tuning.

## Identified Limitations

### Probe comparison is not based on a shared loss baseline

The removed probe implementation recorded each candidate's own first observed loss and normalized subsequent results against it. Candidate scores therefore compared improvement relative to different baselines rather than a single pre-optimization objective value. Any future exploration must use one common pre-step baseline.

### Shape awareness does not determine the important parameters

Layer shape and inferred rank affect the observation window, but they do not intelligently determine:

- The learning-rate search range.
- The number or placement of candidates.
- Probe budget versus selected-run budget.
- Decay strength.
- Retry policy.
- Expected convergence speed.

The same three learning-rate multipliers and fixed controller constants are used across layers with very different optimization geometry.

### The controller observes too little of the optimization state

Loss alone does not reveal why an attempt is slow, noisy, saturated, or unstable. The controller does not currently use:

- Gradient norm or gradient noise.
- Parameter update norm relative to parameter norm.
- Effective step size produced by the optimizer.
- Rounding-assignment change rate.
- Distance from discrete rounding decisions.
- Fraction of saturated or clipped quantized values.
- Curvature or response to learning-rate changes.
- Agreement between continuous optimization loss and final discretized reconstruction quality.

This prevents the controller from distinguishing a learning rate that is too small from a layer that is intrinsically difficult, or a productive noisy trajectory from destructive oscillation.

### One policy spans incompatible optimization dynamics

The same outer probing and convergence policy covers original, AdamW, RAdam, and Prodigy optimizers as well as different quantization objectives and formats. Their learning-rate semantics and warmup behavior are not equivalent. In particular, probing an external learning rate for Prodigy does not necessarily carry the same meaning as probing AdamW because Prodigy adapts its own effective step scale.

INT8 rowwise, ConvRot, FP8, MXFP8, and NVFP4 also expose different scale granularity, clipping, rounding, and reconstruction behavior. Normalizing only by the first loss does not make those dynamics interchangeable.

### Probe overhead can starve the selected attempt

The removed probes consumed the same firm per-layer iteration budget as the selected run. They could spend a meaningful portion of the available work and restart the optimizer before the selected attempt. Exploration remains disabled until a continuous, budget-invariant design is validated.

### Recovery detects catastrophe rather than bad tuning

A retry is recommended for non-finite loss or sustained regression above four times the initial loss. It is not triggered by:

- A learning rate that is stable but far too low.
- Slow convergence relative to the layer's observed response.
- An initially promising candidate that stalls prematurely.
- A selected candidate that underperforms another probe in absolute loss.
- Poor final discrete rounding despite a favorable continuous loss.

The retry learning rate is also selected by a fixed fallback rule rather than a diagnosis of the failed trajectory.

### Best-state capture is coupled to a noise threshold

The controller returns a meaningful-improvement decision to the optimizer loops. Although it records a lower numerical best loss internally, optimizer loops may use the meaningful-improvement result when deciding whether to capture their best tensor or factor state. A genuine improvement smaller than the noise-derived threshold can therefore fail to become the returned learned state.

Best-state capture should depend on strict finite objective improvement, with an optional tolerance only for numerical equality. Plateau detection and scheduling can continue to use noise-aware significance independently.

### Constants are not calibrated against representative data

Values such as `1e-6`, `3 * MAD`, four-times regression, `0.8` and `0.5` decay factors, and the required counts of plateau and stable windows are fixed policy choices. Existing unit tests validate bounded windows, reporting, budget handling, retry mechanics, and routing. They do not demonstrate that these constants minimize reconstruction error, restart frequency, or quantization-quality regressions over representative model layer distributions.

### Diagnostics are useful but insufficient for learning a better policy

The JSON report records attempts, loss windows, learning-rate events, stop reasons, and selected settings. It lacks the optimizer-state and rounding-state signals needed to explain why a setting succeeded or failed. It also does not record a manual baseline or simple-quantization baseline for judging whether learned rounding delivered worthwhile improvement.

## Recommended Redesign

The next controller should remain bounded, deterministic when requested, scheduler-neutral, and compatible with every existing learned route. It should be developed behind the existing opt-in auto-tune interface rather than by rewriting the quantization paths.

### 1. Establish comparable baselines

- Evaluate and record the objective before the first optimization step.
- Restore identical initial learned parameters and random state for every probe.
- Score every candidate against the same pre-step baseline and the same target.
- Record absolute best loss, relative gain, and final discretized reconstruction error.
- Keep strict best-state capture separate from statistical scheduling decisions.

### 2. Instrument optimizer and rounding behavior

Expose a small, format-neutral observation record from learned loops containing:

- Current and best loss.
- Gradient norm.
- Update norm and update-to-parameter ratio.
- Current effective learning rate.
- Non-finite and clipping indicators.
- Rounding-assignment change rate.
- Distance-to-discrete summary, such as mean and upper quantiles.
- Optional format-specific saturation statistics.

The convergence controller should consume observations without owning quantization math, serialization, or format routing.

### 3. Replace equal fixed probes with staged exploration

- Begin with a short shared-baseline response test around the user anchor.
- Reject candidates immediately for non-finite results, destructive loss growth, or excessive update ratios.
- Promote only promising candidates to a longer observation stage.
- Expand upward or downward only when the observed slope and update scale indicate that the initial range missed a useful region.
- Treat optimizer families separately. Prodigy should use signals appropriate to its adapted effective step rather than blindly sharing the Adam-style multiplier policy.
- Reserve a minimum majority of the budget for the selected attempt.

The first version should use explicit, inspectable rules. Learned cross-layer priors can be considered only after sufficient representative tuning reports exist.

### 4. Make convergence diagnosis state-aware

Classify trajectories into at least the following actionable states:

- Productive convergence: continue while marginal gain justifies the cost.
- Stable but step-limited: increase or preserve effective step size rather than decaying it.
- Oscillating: reduce step size based on update and loss behavior.
- Rounding-frozen: stop or change the relaxation policy when assignments no longer move.
- Continuous-only improvement: verify whether discrete reconstruction also improves.
- Diverging or non-finite: restore the best finite state and retry with a diagnosed safer setting.

Stopping should consider convergence rate, uncertainty, discrete-state stability, remaining budget, and the value of further improvement. It should not require the same number of windows for every layer.

### 5. Use layer and format context without changing contracts

Context may include matrix dimensions, aspect ratio, rank, tensor statistics, quantization format, scale granularity, ConvRot status, optimizer family, and objective normalization. These inputs may influence tuning policy only. They must not alter saved tensor names, metadata, scale shapes, or routing decisions.

Reports from earlier layers may provide bounded priors for later layers in the same family, but every layer must retain a safe fallback to the user anchor and must be independently validated.

### 6. Preserve manual control and safe fallback

- Keep auto-tune opt-in while it is experimental.
- Preserve all existing manual learning-rate, schedule, and early-stop behavior when auto-tune is disabled.
- Restore the best finite state if exploration or the selected attempt fails.
- Fall back to the user-provided configuration when evidence is insufficient.
- Report the chosen policy, evidence, consumed budget, fallback reason, and final quality measurements.

## Comfy-Kitchen Applicability

Comfy-kitchen and auto-tune redesign are separate workstreams.

For simple INT8, comfy-kitchen's rowwise quantization and simple dequantization may be useful backend operations if parity is established. For learned rowwise INT8, they may serve as an initial qdata and scale provider or as an independent parity oracle.

They should not replace the learned core, which requires continuous rounding variables, gradients, SVD or other parameterization state, adaptive convergence decisions, explicit-scale requantization, and best-state restoration.

The complete comfy-kitchen ConvRot weight helper is not an appropriate foundation for learned ConvRot when it returns only final qdata and scales. Learned ConvRot needs intermediate floating-point rotated weights, correspondingly rotated calibration activations, the unquantized target, and bias correction in the rotated basis. Individual compatible primitives may be evaluated, but the existing preparation and learned route must remain intact.

Before any backend substitution, compare qdata, scale shapes, dequantized weights, clipping behavior, odd dimensions, device and dtype behavior, and final serialized metadata. A backend mismatch must fall back to the existing local implementation rather than changing the output contract.

## Evaluation Plan

A replacement auto-tune policy should not become the default based only on synthetic convergence tests. Evaluation must include representative layers from real models and compare against both current auto-tune and strong manual settings.

Cover at least:

- INT8 rowwise learned rounding.
- INT8 ConvRot learned rounding.
- Legacy scalar INT8 loading and characterization where applicable.
- FP8 learned routes.
- MXFP8 learned routes.
- NVFP4 learned routes.
- Original, AdamW, RAdam, and Prodigy where each is supported.
- Square, tall, wide, small-rank, large-rank, and unusually difficult layers.
- Stable, noisy, slowly converging, oscillating, and non-finite trajectories.

Measure:

- Final reconstruction or calibration objective in the exact output space used by the converter.
- Final discrete quantized error, not only continuous relaxation loss.
- Best result relative to a strong manual configuration.
- Iterations and wall-clock time.
- Number of retries or user restarts required.
- Frequency and magnitude of quality regressions.
- Budget consumed by exploration versus final optimization.
- Determinism under a fixed seed.

Acceptance should require no material loss of quality against tuned manual baselines, a substantial reduction in restart frequency, bounded exploration overhead, and no change to saved-model compatibility. Results should be reported by format, optimizer, layer family, and shape rather than only as a global average.

## Recommended Work Order

1. Correct the terminology in user documentation so the existing feature is described as heuristic adaptive tuning.
2. Add shared pre-step baselines and decouple strict best-state capture from meaningful-improvement scheduling.
3. Extend tuning reports with optimizer and rounding-state observations without changing output formats.
4. Build a representative layer corpus and benchmark current manual and automatic behavior.
5. Introduce staged candidate exploration behind an experimental policy selector.
6. Add state-aware convergence and recovery after the observation data validates its thresholds.
7. Evaluate any comfy-kitchen primitives independently from auto-tune changes.
8. Consider changing defaults only after compatibility and quality acceptance criteria are met.

## Conclusion

The corrected implementation is a useful deterministic convergence controller, but it is not yet an intelligent tuner. Its shape-aware windows, bounded budget, continuous optimizer state, reporting, and recovery provide a foundation. The next step is to improve the information available to the controller before reintroducing any learning-rate exploration, not to broaden backend replacement or disturb established learned and simple quantization contracts.
