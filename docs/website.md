# Website

The public project site is built from Markdown with MkDocs Material and published to GitHub Pages by `.github/workflows/pages.yml`.

## Content policy

Keep stable technical explanations in the main documentation tree. Put dated project updates in `docs/blog/posts/` with front matter containing at least `date`, `categories`, and `authors`. Do not put generated datasets or unreviewed model output into the navigation.

The Japan track must retain its scientific-research-only restriction. Journal posts may summarize it, but must not present exploratory anomalies as earthquake prediction.

## Local preview

Install the site dependencies in the project virtual environment, then run `mkdocs serve`. Before publishing, run `mkdocs build --strict` so broken navigation and Markdown references fail the build.

## Publish

1. Push the site changes to the repository's `main` branch.
2. In GitHub, open **Settings > Pages** and set **Source** to **GitHub Actions**.
3. Open **Actions > Publish documentation** and monitor the workflow. It runs automatically for changes under `docs/`, `mkdocs.yml`, `requirements-site.txt`, or the workflow file itself.

The published site is expected at `https://dannyayers.github.io/elfquake/`. The workflow can also be started manually from **Actions > Publish documentation > Run workflow**.
