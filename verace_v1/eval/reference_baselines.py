"""
Reference baseline benchmark scores for small "Thinking" models, hardcoded from the
public MiniCPM5-1B evaluation leaderboard (source: public_leaderboard_en.png,
"Evaluation Results of MiniCPM5-1B"). Not computed -- these are published numbers for
other models, kept here so Verace V1's own benchmark_runner.py results can be compared
against comparably-sized published models once real (trained-checkpoint) eval numbers
exist for Verace V1.
"""
from typing import Dict, List

REFERENCE_MODELS: List[str] = [
    "MiniCPM5-1B(Thinking)",
    "Qwen3-0.6B(Thinking)",
    "Qwen3.5-0.8B(Thinking)",
    "LFM2.5-1.2B(Thinking)",
]

# category -> ordered list of benchmark names in that category
REFERENCE_CATEGORIES: Dict[str, List[str]] = {
    "General Knowledge": ["MMLU-Pro", "MMLU-Redux"],
    "Domain-Specific Knowledge": ["GPQA-Diamond", "SuperGPQA"],
    "Coding & Programming": ["LCB-Pro 25Q2(Easy)", "OJBench", "LCB-v6(@Avg3)"],
    "Instruction Following": ["IFBench", "IFEval", "Multi-IF", "MultiChallenge"],
    "Mathematical Reasoning": ["AIME-2025(@Avg16)", "AIME-2026(@Avg16)", "HMMT Feb 2026(@Avg16)", "MATH-500"],
    "Logical Reasoning": ["BBH", "BBEH"],
    "Agentic Evaluation": ["BFCLv4", "τ²-Bench Telecom-AA"],
}

# benchmark name -> {model name -> score}. "Average Score" has no category (matches source table).
REFERENCE_SCORES: Dict[str, Dict[str, float]] = {
    "Average Score": {
        "MiniCPM5-1B(Thinking)": 42.57,
        "Qwen3-0.6B(Thinking)": 26.77,
        "Qwen3.5-0.8B(Thinking)": 25.14,
        "LFM2.5-1.2B(Thinking)": 35.61,
    },
    "MMLU-Pro": {
        "MiniCPM5-1B(Thinking)": 48.85,
        "Qwen3-0.6B(Thinking)": 35.63,
        "Qwen3.5-0.8B(Thinking)": 42.74,
        "LFM2.5-1.2B(Thinking)": 47.98,
    },
    "MMLU-Redux": {
        "MiniCPM5-1B(Thinking)": 70.06,
        "Qwen3-0.6B(Thinking)": 55.47,
        "Qwen3.5-0.8B(Thinking)": 61.50,
        "LFM2.5-1.2B(Thinking)": 66.08,
    },
    "GPQA-Diamond": {
        "MiniCPM5-1B(Thinking)": 26.26,
        "Qwen3-0.6B(Thinking)": 25.42,
        "Qwen3.5-0.8B(Thinking)": 30.98,
        "LFM2.5-1.2B(Thinking)": 34.85,
    },
    "SuperGPQA": {
        "MiniCPM5-1B(Thinking)": 23.14,
        "Qwen3-0.6B(Thinking)": 20.79,
        "Qwen3.5-0.8B(Thinking)": 22.92,
        "LFM2.5-1.2B(Thinking)": 22.83,
    },
    "LCB-Pro 25Q2(Easy)": {
        "MiniCPM5-1B(Thinking)": 22.68,
        "Qwen3-0.6B(Thinking)": 4.12,
        "Qwen3.5-0.8B(Thinking)": 0.00,
        "LFM2.5-1.2B(Thinking)": 6.19,
    },
    "OJBench": {
        "MiniCPM5-1B(Thinking)": 7.33,
        "Qwen3-0.6B(Thinking)": 0.86,
        "Qwen3.5-0.8B(Thinking)": 0.43,
        "LFM2.5-1.2B(Thinking)": 1.94,
    },
    "LCB-v6(@Avg3)": {
        "MiniCPM5-1B(Thinking)": 33.52,
        "Qwen3-0.6B(Thinking)": 16.00,
        "Qwen3.5-0.8B(Thinking)": 5.33,
        "LFM2.5-1.2B(Thinking)": 21.33,
    },
    "IFBench": {
        "MiniCPM5-1B(Thinking)": 46.67,
        "Qwen3-0.6B(Thinking)": 25.67,
        "Qwen3.5-0.8B(Thinking)": 29.33,
        "LFM2.5-1.2B(Thinking)": 41.67,
    },
    "IFEval": {
        "MiniCPM5-1B(Thinking)": 80.41,
        "Qwen3-0.6B(Thinking)": 59.89,
        "Qwen3.5-0.8B(Thinking)": 59.89,
        "LFM2.5-1.2B(Thinking)": 84.84,
    },
    "Multi-IF": {
        "MiniCPM5-1B(Thinking)": 43.54,
        "Qwen3-0.6B(Thinking)": 36.56,
        "Qwen3.5-0.8B(Thinking)": 32.31,
        "LFM2.5-1.2B(Thinking)": 55.61,
    },
    "MultiChallenge": {
        "MiniCPM5-1B(Thinking)": 19.48,
        "Qwen3-0.6B(Thinking)": 18.97,
        "Qwen3.5-0.8B(Thinking)": 23.97,
        "LFM2.5-1.2B(Thinking)": 23.28,
    },
    "AIME-2025(@Avg16)": {
        "MiniCPM5-1B(Thinking)": 40.42,
        "Qwen3-0.6B(Thinking)": 16.25,
        "Qwen3.5-0.8B(Thinking)": 1.04,
        "LFM2.5-1.2B(Thinking)": 31.88,
    },
    "AIME-2026(@Avg16)": {
        "MiniCPM5-1B(Thinking)": 40.42,
        "Qwen3-0.6B(Thinking)": 12.29,
        "Qwen3.5-0.8B(Thinking)": 0.21,
        "LFM2.5-1.2B(Thinking)": 31.67,
    },
    "HMMT Feb 2026(@Avg16)": {
        "MiniCPM5-1B(Thinking)": 25.76,
        "Qwen3-0.6B(Thinking)": 9.85,
        "Qwen3.5-0.8B(Thinking)": 0.57,
        "LFM2.5-1.2B(Thinking)": 21.21,
    },
    "MATH-500": {
        "MiniCPM5-1B(Thinking)": 91.60,
        "Qwen3-0.6B(Thinking)": 72.60,
        "Qwen3.5-0.8B(Thinking)": 30.40,
        "LFM2.5-1.2B(Thinking)": 89.00,
    },
    "BBH": {
        "MiniCPM5-1B(Thinking)": 71.89,
        "Qwen3-0.6B(Thinking)": 47.86,
        "Qwen3.5-0.8B(Thinking)": 54.58,
        "LFM2.5-1.2B(Thinking)": 57.32,
    },
    "BBEH": {
        "MiniCPM5-1B(Thinking)": 12.14,
        "Qwen3-0.6B(Thinking)": 3.78,
        "Qwen3.5-0.8B(Thinking)": 8.53,
        "LFM2.5-1.2B(Thinking)": 8.64,
    },
    "BFCLv4": {
        "MiniCPM5-1B(Thinking)": 25.15,
        "Qwen3-0.6B(Thinking)": 25.43,
        "Qwen3.5-0.8B(Thinking)": 25.53,
        "LFM2.5-1.2B(Thinking)": 10.60,
    },
    "τ²-Bench Telecom-AA": {
        "MiniCPM5-1B(Thinking)": 79.53,
        "Qwen3-0.6B(Thinking)": 21.10,
        "Qwen3.5-0.8B(Thinking)": 47.70,
        "LFM2.5-1.2B(Thinking)": 19.60,
    },
}


def print_reference_leaderboard() -> None:
    """
    Renders REFERENCE_SCORES as a grouped terminal table matching the source
    leaderboard image exactly: category-grouped rows, MiniCPM5-1B(Thinking) as the
    highlighted column (pale blue always, bold bright blue when it's the best score
    in that row -- same convention the source image uses). Falls back to a plain
    ASCII table if `rich` isn't installed.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        _print_reference_leaderboard_plain()
        return

    focus_model = REFERENCE_MODELS[0]  # "MiniCPM5-1B(Thinking)"
    other_models = REFERENCE_MODELS[1:]

    table = Table(show_lines=False, header_style="bold")
    table.add_column("Category", style="bold", no_wrap=True)
    table.add_column("Benchmark", no_wrap=True)
    table.add_column(focus_model, justify="right", no_wrap=True)
    for m in other_models:
        table.add_column(m, justify="right", no_wrap=True)

    def _cell(name: str, model: str) -> Text:
        score = REFERENCE_SCORES[name][model]
        text = Text(f"{score:.2f}", justify="right")
        if model == focus_model:
            row_best = max(REFERENCE_SCORES[name].values())
            if score == row_best:
                text.stylize("bold black on bright_blue")
            else:
                text.stylize("black on #EAF3FF")
        return text

    avg_row = ["", Text("Average Score", style="bold")]
    avg_row += [_cell("Average Score", m) for m in REFERENCE_MODELS]
    table.add_row(*avg_row, end_section=True)

    categories = list(REFERENCE_CATEGORIES.items())
    for cat_idx, (category, names) in enumerate(categories):
        for i, name in enumerate(names):
            row = [category if i == 0 else "", name]
            row += [_cell(name, m) for m in REFERENCE_MODELS]
            table.add_row(*row, end_section=(i == len(names) - 1 and cat_idx < len(categories) - 1))

    Console(width=140).print(table)


def _print_reference_leaderboard_plain() -> None:
    """ASCII fallback for print_reference_leaderboard() when `rich` isn't installed."""
    header = f"{'Category':<26}{'Benchmark':<22}" + "".join(f"{m:>22}" for m in REFERENCE_MODELS)
    print(header)
    print("-" * len(header))

    def _row(category: str, name: str) -> str:
        cells = "".join(f"{REFERENCE_SCORES[name][m]:>22.2f}" for m in REFERENCE_MODELS)
        return f"{category:<26}{name:<22}{cells}"

    print(_row("", "Average Score"))
    for category, names in REFERENCE_CATEGORIES.items():
        for i, name in enumerate(names):
            print(_row(category if i == 0 else "", name))


def print_comparison_table(verace_scores: Dict[str, float], verace_label: str = "Verace-V1") -> None:
    """
    Prints REFERENCE_SCORES side by side with a Verace V1 result dict (same keys as
    REFERENCE_SCORES, e.g. from repeated VeraceV1Evaluator.evaluate_benchmark() calls),
    grouped by category, so a real Verace V1 eval run can be read against these
    published baselines at a glance.
    """
    models = [verace_label] + REFERENCE_MODELS
    header = f"{'Benchmark':<24}" + "".join(f"{m:>22}" for m in models)
    print(header)
    print("-" * len(header))

    def _row(name: str, scores: Dict[str, float]) -> str:
        cells = [f"{scores.get(verace_label, float('nan')):>22.2f}"]
        cells += [f"{REFERENCE_SCORES[name].get(m, float('nan')):>22.2f}" for m in REFERENCE_MODELS]
        return f"{name:<24}" + "".join(cells)

    print(_row("Average Score", {verace_label: verace_scores.get("Average Score", float("nan"))}))
    for category, names in REFERENCE_CATEGORIES.items():
        print(f"\n{category}")
        for name in names:
            print(_row(name, {verace_label: verace_scores.get(name, float("nan"))}))
