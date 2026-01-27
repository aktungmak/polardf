#!/usr/bin/env python3
"""
Parse unittest output and update README.md with a W3C test suite report.

Usage:
    python scripts/update_test_report.py <test_output_file>

The script reads unittest verbose output, extracts pass/fail/skip counts,
and appends or updates a summary section at the bottom of README.md.
"""

import re
import sys
from datetime import datetime
from pathlib import Path


def parse_unittest_output(output: str) -> dict:
    """Parse unittest summary output and return counts by suite.

    Parses the summary lines at the end of each test run:
    - "Ran X tests in Y.YYYs"
    - "OK" / "OK (skipped=N)" / "FAILED (failures=X, errors=Y, skipped=Z)"
    """
    results = {
        "sparql10": {"passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        "sparql11": {"passed": 0, "failed": 0, "skipped": 0, "errors": 0},
    }

    current_suite = None
    total_tests = 0

    for line in output.splitlines():
        # Detect which suite we're in from the unittest command line
        if "tests.test_w3c_sparql10" in line:
            current_suite = "sparql10"
        elif "tests.test_w3c_sparql11" in line:
            current_suite = "sparql11"

        if current_suite is None:
            continue

        # Parse "Ran X tests in Y.YYYs"
        if match := re.match(r"Ran (\d+) tests? in", line):
            total_tests = int(match.group(1))

        # Parse result line: OK or FAILED with optional details
        # Format: "OK", "OK (skipped=N)", "FAILED (failures=X, errors=Y, skipped=Z)"
        if line.startswith("OK") or line.startswith("FAILED"):
            failures = errors = skipped = 0

            if match := re.search(r"failures=(\d+)", line):
                failures = int(match.group(1))
            if match := re.search(r"errors=(\d+)", line):
                errors = int(match.group(1))
            if match := re.search(r"skipped=(\d+)", line):
                skipped = int(match.group(1))

            passed = total_tests - failures - errors - skipped

            results[current_suite] = {
                "passed": passed,
                "failed": failures,
                "errors": errors,
                "skipped": skipped,
            }
            # Reset for next suite
            total_tests = 0

    return results


def generate_report(results: dict) -> str:
    """Generate markdown report section."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "",
        "### W3C Test Suite Results",
        "",
        f"_Last updated: {timestamp}_",
        "",
        "| Suite | Passed | Failed | Errors | Skipped | Total |",
        "|-------|--------|--------|--------|---------|-------|",
    ]

    total_passed = total_failed = total_errors = total_skipped = 0

    for suite_name, counts in sorted(results.items()):
        passed = counts["passed"]
        failed = counts["failed"]
        errors = counts["errors"]
        skipped = counts["skipped"]
        total = passed + failed + errors + skipped

        total_passed += passed
        total_failed += failed
        total_errors += errors
        total_skipped += skipped

        display_name = "SPARQL 1.0" if suite_name == "sparql10" else "SPARQL 1.1"
        lines.append(
            f"| {display_name} | {passed} | {failed} | {errors} | {skipped} | {total} |"
        )

    grand_total = total_passed + total_failed + total_errors + total_skipped
    lines.append(
        f"| **Total** | **{total_passed}** | **{total_failed}** | **{total_errors}** | **{total_skipped}** | **{grand_total}** |"
    )

    # Calculate pass rate
    if grand_total > 0:
        pass_rate = (total_passed / grand_total) * 100
        lines.append("")
        lines.append(
            f"**Pass rate: {pass_rate:.1f}%** ({total_passed}/{grand_total} tests)"
        )

    return "\n".join(lines)


def update_readme(report: str) -> None:
    """Update README.md with the test report section."""
    readme_path = Path(__file__).parent.parent / "README.md"

    content = readme_path.read_text()

    # Remove existing report section if present
    marker = "### W3C Test Suite Results"
    if marker in content:
        idx = content.index(marker)
        # Find where to cut - go back to include the preceding newline
        while idx > 0 and content[idx - 1] == "\n":
            idx -= 1
        content = content[:idx]

    # Append new report
    content = content.rstrip() + "\n" + report + "\n"

    readme_path.write_text(content)
    print(f"Updated {readme_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <test_output_file>", file=sys.stderr)
        sys.exit(1)

    output_file = Path(sys.argv[1])
    if not output_file.exists():
        print(f"Error: {output_file} not found", file=sys.stderr)
        sys.exit(1)

    output = output_file.read_text()
    results = parse_unittest_output(output)
    report = generate_report(results)
    update_readme(report)

    # Print summary
    for suite, counts in sorted(results.items()):
        total = sum(counts.values())
        print(
            f"{suite}: {counts['passed']} passed, {counts['failed']} failed, "
            f"{counts['errors']} errors, {counts['skipped']} skipped ({total} total)"
        )


if __name__ == "__main__":
    main()
