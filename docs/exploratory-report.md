# Exploratory Report

The first report should validate data usability, not model performance.

## Required Sections

* dataset version and source requests
* record counts by month
* magnitude distribution
* depth distribution
* map or regional count summary
* missing fields and invalid values
* duplicate event identifiers
* timestamp precision and timezone checks
* known source or connector failures

## Outputs

Save the report as a short Markdown or notebook artifact with generated tables and plots. Include enough metadata to reproduce the source pull and normalization run.

Current report: [Smoke Exploratory Report](exploratory-report-smoke.md).

## Pass Criteria

The dataset is ready for baseline modeling only if event times, locations, magnitudes, and region labels are complete enough for the pilot scope.
