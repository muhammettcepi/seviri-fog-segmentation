# Coding guidelines

The goal: code that a new reader understands on the first pass. When short and clever competes with longer
and plain, choose longer and plain.

These guidelines take their rules from four public sources and add the conventions this codebase already
follows. Ruff checks what can be checked (see [Tooling](#tooling)); the rest is for reviews.

| Source | What we take from it |
|---|---|
| [PEP 20, The Zen of Python](https://peps.python.org/pep-0020/) | The principles below |
| [PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257/) | Layout, naming and docstrings; `ruff format` applies the layout |
| [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) | Its decisions on comprehensions, conditional expressions, "power features" and function length |
| [Scientific Python Development Guide](https://learn.scientific-python.org/development/guides/style/) | The Ruff rule groups |

## Principles

From PEP 20:

> Explicit is better than implicit.
> Simple is better than complex.
> Flat is better than nested.
> Sparse is better than dense.
> Readability counts.
> If the implementation is hard to explain, it's a bad idea.

PEP 8 gives the reason: code is read much more often than it is written.

## Writing simple code

**One function, one job.** If a function checks its inputs, computes something and logs the result, those
are three functions. A function longer than about 40 lines (Google 3.18) or with more than 10 branches
(Ruff `C901`) should be split into named steps. `cmd_evaluate` in `cli.py` shows the pattern: the command
reads as a list of steps (`check_evaluate_inputs`, `check_predictions_match`, `log_comparisons`,
`held_out_metrics`), and each step can be read and tested on its own.

**Comprehensions only for simple cases** (Google 2.7). One `for` and at most one `if`. Otherwise write
the loop:

```python
# Yes
names = [station for station in stations if station.startswith("06")]

# No: two for clauses and a filter in one expression
pairs = [(s, t) for s in stations for t in scene_times if near(s, t)]

# Yes: the same as a loop
pairs = []
for station in stations:
    for time in scene_times:
        if near(station, time):
            pairs.append((station, time))
```

**Conditional expressions only when they fit on one line and read as a sentence** (Google 2.11):
`label = "fog" if is_fog else "clear"`. Anything longer is an `if` statement.

**No power features** (Google 2.19): no metaclasses, `getattr` tricks, monkey-patching in library code, or
import hacks. They make code shorter and much harder to follow.

**Flat, not nested.** Return early instead of nesting `if` inside `if`. Raise on bad input at the top of a
function (`check_*` helpers), then let the main path run without indentation.

**Explicit arguments.** Pass data, settings and parameters as arguments rather than reading globals or
bundling everything into one context object. Pipeline functions may take up to 10 arguments; more than
that means some of them belong together in a frozen dataclass (as `LabelParams` and `TeacherParams` do).

**Name the intermediate values.** A named variable explains a step better than a comment, and better than
a chained one-liner:

```python
# Yes
is_close = located["distance_km"] <= max_distance_km
is_low = (located["elevation_m"] <= max_elevation_m) | located["elevation_m"].isna()
return located[is_close & is_low]
```

## Names

- Functions that **return** something are named after what they return: `fls_truth`, `station_cases`,
  `season_bounds`. Functions that **do** something start with a verb: `download`, `write_atomically`.
  Functions that only raise on bad input start with `check_`.
- **Units go in the name**: `visibility_m`, `max_distance_km`, `scan_delay_min`, `EARTH_RADIUS_KM`.
  A number without a unit in its name is a bug waiting to happen.
- Booleans read as a yes/no question: `is_fog`, `has_case`, `hidden`.
- Constants are `UPPER_CASE` at the top of the module. Numbers that are settings belong in
  `configs/default.yaml`, not in the code.
- Conventional import names: `np`, `pd`, `xr`, `mpl`, `plt` (Ruff `ICN`).

## Docstrings and comments

- Every public function has a docstring (PEP 257). The first line says **what the function returns or
  does**, in plain English, ending with a full stop: `"""Great-circle distance in km from every (lon, lat)
  to one point."""`
- Add more lines only for what the signature cannot say: shapes of arrays, units, what `NaN` means,
  which input wins when two disagree.
- Comments say **why**, not what: `# SEVIRI scans Central Europe about 12 minutes after the slot time`.
  If a comment only repeats the code, delete it.
- A `# noqa` always names the rule and gives the reason:
  `# noqa: ARG002 - same signature as MlflowTracker.run`.

## Errors and logging

- An error message says what is wrong **and how to fix it**:
  ``f"{path} not found; run `fogseg train` first"``.
- Raise the most specific built-in exception (`FileNotFoundError`, `ValueError`, `PermissionError`).
- Library code logs with `log = logging.getLogger("fogseg")` and never uses `print` (Ruff `T20`).
  Notebooks and figure scripts may print.
- Log with `%`-style arguments, not f-strings: `log.info("Wrote %d cases to %s", n, path)` (Ruff `G`).

## Files and data

- Paths are `pathlib.Path` objects (Ruff `PTH`).
- Write results atomically: write `name.part`, then `part.replace(path)`. A crash then never leaves a half
  file that a rerun would mistake for a finished one.
- `zip()` always says `strict=True`, so sequences of different lengths fail loudly instead of being
  silently cut short (Ruff `B905`).

## Tests

- Test names read as a sentence about behaviour: `test_deduplicate_prefers_a_known_truth_over_the_source`.
- One assertion per line, so a failure points at the exact condition (Ruff `PT018`).
- `pytest.raises` always has a `match=`, and the block holds only the call that should raise (Ruff
  `PT011`, `PT012`). Use a raw string for the pattern, and escape a literal dot: `match=r"labels\.zarr"`.
- A bug fix comes with a test that failed before the fix.

## Tooling

```bash
ruff check .            # lint: the rules below
ruff format .           # layout: PEP 8, lines up to 110 characters
pytest                  # tests
```

CI runs `ruff check` and `ruff format --check` on every push and pull request; both must pass. Ruff is
pinned in `pyproject.toml`, so local runs and CI agree.

### Ruff rules

Switched on in `pyproject.toml`, on top of Ruff's defaults:

| Group | Checks |
|---|---|
| `C90`, `PL` | size limits: complexity 10, branches 12, statements 50, returns 6, arguments 10 |
| `B`, `PIE`, `RUF`, `TRY` | bug-prone patterns, unnecessary code, exception handling |
| `SIM`, `RET`, `C4`, `FURB` | simpler conditions, returns and comprehensions |
| `ARG` | unused arguments |
| `PT` | pytest style |
| `PTH` | `pathlib` instead of `os.path` |
| `T10`, `T20` | no debugger or `print` left in library code |
| `G`, `LOG` | logging |
| `ICN`, `NPY`, `PD` | conventional imports, NumPy and pandas pitfalls |
| `UP`, `PGH`, `RSE`, `EXE` | modern syntax, specific `noqa` codes, raise style, executable files |

Switched off on purpose:

| Rule | Why |
|---|---|
| `TRY003`, and `EM` not selected | Long error messages that say how to fix the problem are wanted here |
| `PLC0415` | `cli.py` imports heavy libraries inside each command so `fogseg --help` starts fast |
| `PLR2004` | Thresholds and physical constants read better where they are used |
| `PD011` | `.values` is right for xarray, which this rule mistakes for pandas |
| `PD101` | `nunique() == 1` says "all values are equal" more plainly than the suggested rewrite |
| `TC` not selected | Moving imports into `TYPE_CHECKING` blocks everywhere adds noise for no reader benefit |
| `ARG`, `PLR0913` in tests | Fakes must match the signature of what they replace |

## When in doubt

Look at the code around you and do the same (Google, "Be consistent"). If the surrounding code breaks
these guidelines, fix it in its own commit rather than copying the problem.
