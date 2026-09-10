"""CLI entry point for the AI Resume Screening & Ranking System."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE
from src.models import PipelineOutput
from src.pipeline import run_pipeline


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_results(output: PipelineOutput, output_path: Path):
    """Save pipeline output to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [r.model_dump() for r in output.results],
        "rejected": [r.model_dump() for r in output.rejected],
        "summary": output.summary.model_dump(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"Results saved to {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input directory containing resume files (default: ./resumes)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path (default: ./output/results.json)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM-assisted extraction and scoring (requires API key)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate input
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.is_dir():
        print(f"Error: Input path is not a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"LLM:    {'enabled' if args.use_llm else 'disabled'}")
    print()

    output = run_pipeline(args.input, args.output, use_llm=args.use_llm)

    # Save results
    save_results(output, args.output)

    # Print summary
    print("\n=== Pipeline Summary ===")
    print(f"Total resumes:      {output.summary.total_resumes}")
    print(f"Successfully parsed: {output.summary.successfully_parsed}")
    print(f"Eligible:           {output.summary.eligible}")
    print(f"Rejected:           {output.summary.rejected}")
    print(f"Failed/Unreadable:  {output.summary.failed_unreadable}")
    print()

    if output.results:
        print("=== Top Candidates ===")
        for r in output.results[:10]:
            print(f"  #{r.rank:2d} | {r.total_score:5.1f} | {r.candidate_name}")
    print()


if __name__ == "__main__":
    main()
