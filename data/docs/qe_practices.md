# Quality Engineering for RAG Systems

Quality Engineering (QE) for RAG systems means treating evaluation as a
first-class, automated, continuous activity rather than a one-off manual check.
Because language models are non-deterministic and their behavior drifts as
models and data change, RAG pipelines need the same regression safety net that
traditional software enjoys.

## Golden datasets

A golden dataset is a curated set of questions paired with known-good reference
answers. It is the fixed yardstick against which every change to the pipeline is
measured. When a prompt, chunk size, embedding model, or language model changes,
re-running the golden dataset shows whether quality improved or regressed.

## Quality gates

A quality gate is a threshold on an evaluation metric that a build must clear to
be considered acceptable. For example, a team might require mean faithfulness to
stay above 0.70. If a change drops a metric below its gate, the build fails,
which prevents a quality regression from reaching production.

## Continuous evaluation in CI

Wiring evaluation into continuous integration means every pull request is scored
automatically. Fast, deterministic unit tests verify the plumbing on every
commit, while a heavier live evaluation against the golden dataset runs on a
schedule or before release. Separating the two keeps feedback fast without
sacrificing depth.

## Choosing the judge model

When a language model scores another model's output, the judge must be capable
enough to follow structured-output instructions reliably. A judge that is too
small may return malformed output and produce noisy scores, so the judge model
is itself a quality decision worth validating.
