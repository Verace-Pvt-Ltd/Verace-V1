"""
Unit tests for the TinyStories-paper-methodology GPT-4 evaluation harness.
No API calls -- these test the prompt set and the grading-response parser only.
"""
from verace_v1.eval.tinystories_gpt4_eval import EVAL_PROMPTS, PromptResult, CompletionGrade, _parse_grade, summarize_results


def test_eval_prompts_are_50_unique_nonempty():
    assert len(EVAL_PROMPTS) == 50
    assert len(set(EVAL_PROMPTS)) == 50
    assert all(isinstance(p, str) and len(p.strip()) > 0 for p in EVAL_PROMPTS)


def test_parse_grade_exact_paper_format():
    grade = _parse_grade("Grammar: 8/10, Creativity: 7/10, Consistency: 7/10, Age group: E")
    assert grade == CompletionGrade(grammar=8.0, creativity=7.0, consistency=7.0, age_group="E")


def test_parse_grade_tolerates_surrounding_prose():
    grade = _parse_grade(
        "This completion is grammatically solid and stays on topic.\n\n"
        "Grammar: 9/10, Creativity: 6/10, Consistency: 8/10, Age group: C\n"
    )
    assert grade is not None
    assert grade.grammar == 9.0 and grade.creativity == 6.0 and grade.consistency == 8.0 and grade.age_group == "C"


def test_parse_grade_returns_none_on_malformed_response():
    assert _parse_grade("I cannot grade this completion.") is None
    assert _parse_grade("") is None


def test_summarize_results_averages_per_prompt_then_across_prompts():
    results = [
        PromptResult(
            prompt="p1", completions=["c1", "c2"],
            grades=[
                CompletionGrade(grammar=8, creativity=6, consistency=7, age_group="D"),
                CompletionGrade(grammar=6, creativity=8, consistency=5, age_group="D"),
            ]
        ),
        PromptResult(
            prompt="p2", completions=["c1"],
            grades=[CompletionGrade(grammar=10, creativity=10, consistency=10, age_group="F")]
        ),
        PromptResult(prompt="p3_ungraded", completions=["c1"], grades=[]),  # parsing failed for all completions
    ]
    summary = summarize_results(results)
    # p1 mean: grammar=7, creativity=7, consistency=6; p2: grammar=10, creativity=10, consistency=10
    assert abs(summary["grammar"] - 8.5) < 1e-9
    assert abs(summary["creativity"] - 8.5) < 1e-9
    assert abs(summary["consistency"] - 8.0) < 1e-9
    assert summary["prompts_graded"] == 2
    assert summary["prompts_total"] == 3


if __name__ == "__main__":
    test_eval_prompts_are_50_unique_nonempty()
    test_parse_grade_exact_paper_format()
    test_parse_grade_tolerates_surrounding_prose()
    test_parse_grade_returns_none_on_malformed_response()
    test_summarize_results_averages_per_prompt_then_across_prompts()
