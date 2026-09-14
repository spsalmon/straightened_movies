# Tests

`test_straightened_movies.py` covers the geometry helpers, the flip-transform
algebra, orientation prediction and movie assembly, plus an end-to-end run of
the script as a subprocess against a synthetic experiment generated at runtime.
No bundled data, no SLURM, no pipeline required.

Run from the repository root:

```bash
~/.local/bin/micromamba run -n towbintools python -m pytest tests/ -v
```

The tests skip themselves if the dependencies are not installed.
