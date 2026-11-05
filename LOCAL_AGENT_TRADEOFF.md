# Local agent: rebuild the paper after the Tier 3 rewrite, and push

Written 2026-11-05 by the reviewing session. One job, under ten minutes, no model runs. The
rules of `LOCAL_AGENT_HANDOFF.md` §2 still apply (sub-branch, merge back with `--no-ff`, never
`main`, never force-push, never hand-edit a result file).

## Why

The paper (`latex/transformer_context_contagion.tex`) was rewritten in iteration 12 against the
final Tier 2 numbers: new abstract, a ranked summary section, rewritten status and claims tables,
and new tables for C4, C5 and C6. The reviewing session cannot compile LaTeX, so the committed
`latex/transformer_context_contagion.pdf` is from your iteration-6 build and is now two
revisions behind the source. You have TeX; rebuild it so the reviewer can read the result.

## Do this

1. Update and branch:
   ```bash
   git fetch origin
   git checkout claude/fervent-ritchie-l5m9uw && git pull --ff-only
   git checkout -b claude/fervent-ritchie-l5m9uw-iter13-pdf
   ```
   Confirm `git log --oneline -1` shows the merge of iteration 12 ("Tier 3 paper rewrite") or later.

2. Build, twice so the cross-references settle, and keep the log:
   ```bash
   cd latex
   latexmk -pdf -interaction=nonstopmode transformer_context_contagion.tex > build.log 2>&1 || true
   latexmk -pdf -interaction=nonstopmode transformer_context_contagion.tex >> build.log 2>&1
   cd ..
   ```
   If `latexmk` is absent, run `pdflatex -interaction=nonstopmode` three times instead.

3. Check the build before committing. All of these must hold; if one fails, do not commit the
   PDF, and put the relevant lines of `latex/build.log` in your report instead:
   - the build exits 0 and `latex/transformer_context_contagion.pdf` has a fresh timestamp;
   - `grep -c "LaTeX Warning: Reference .* undefined" latex/build.log` prints `0`;
   - `grep -c "LaTeX Warning: Citation .* undefined" latex/build.log` prints `0`;
   - `grep -ci "file .* not found" latex/build.log` prints `0` (every figure resolves);
   - `grep -c "^!" latex/build.log` prints `0` (no TeX errors);
   - the first page shows the title, the date line "November 2026 (revised from the June 2026
     progress report)", a boxed revision note, and an abstract that begins "Two companion
     analyses" and ends "counting repeated n-grams." Use `pdftotext -l 1` if available, otherwise
     open it.
   Overfull-hbox warnings are acceptable; everything else above is not.

4. Commit only the PDF (the `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `build.log` files are
   ignored by `latex/.gitignore`; check `git status` shows exactly the one file):
   ```bash
   git add latex/transformer_context_contagion.pdf
   git commit -m "Rebuild the paper PDF after the Tier 3 rewrite (iteration 12)"
   ```

5. Merge and push so the reviewing session can fetch it:
   ```bash
   git push -u origin claude/fervent-ritchie-l5m9uw-iter13-pdf
   git checkout claude/fervent-ritchie-l5m9uw
   git merge --no-ff claude/fervent-ritchie-l5m9uw-iter13-pdf -m "Merge iteration 13: PDF rebuilt after Tier 3"
   git push origin claude/fervent-ritchie-l5m9uw
   ```

6. Report back in a few lines: the commit hash on the parent branch, the page count of the
   PDF, and the output of the five checks in step 3. If any check failed, say which and quote
   the log lines; do not try to fix the source yourself.

## Not in scope

No experiment reruns, no edits to the `.tex`, no changes to `results_log.md`. If the build
reveals a problem in the source, that is for the reviewing session to fix; report it.
