"""
TinyStories paper's GPT-4 evaluation methodology (Eldan & Li, arXiv:2305.07759), adapted
for Verace V1: for each of ~50 manually-constructed story-beginning prompts, generate 10
completions at temperature=1, ask GPT-4 to grade each completion on Grammar, Creativity,
and Consistency (0-10 each) plus a guessed writer age-group, and average the scores.

The paper does not publish its exact 50-prompt list (checked: not in the paper body or
visible appendix), so EVAL_PROMPTS below is a representative set constructed in the same
style/spirit -- simple, ~3-4-year-old vocabulary, room for a natural continuation -- not a
byte-identical reproduction of their set. This means absolute scores are not directly
comparable to the paper's table; the *methodology* (same rubric, same prompt template,
same completions-per-prompt, same aggregation) is what's held constant for a fair-process
comparison, matching what's actually reproducible without the paper's supplementary data.

Requires OPENAI_API_KEY to actually run GPT-4 grading -- this module fails loudly if it's
missing, rather than silently skipping grading (see run_evaluation()'s upfront check).
"""

import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from verace_v1.serving.hyper_generate import VeraceV1Generator

COMPLETIONS_PER_PROMPT = 10
GENERATION_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 200

AGE_GROUPS = {
    "A": "3 or under", "B": "4-5", "C": "6-7", "D": "8-9", "E": "10-12", "F": "13-16",
}

# Representative TinyStories-style story-beginning prompts (see module docstring: the
# paper's own ~50-prompt list isn't published, so this is constructed in the same style,
# not a reproduction of theirs).
EVAL_PROMPTS: List[str] = [
    "Jack was hungry, so he went looking for",
    "Lily wanted to get either a cat or a dog. Her mother didn't let her get a dog so instead she",
    "Once upon a time there was a little girl named Lucy who loved to",
    "The girl said to them, \"Hi, I'm Jane. What are your names?\" They said,",
    "Tom and his sister were playing in the garden when they found a",
    "One sunny day, a small rabbit named Ben went to the park to",
    "Once there was a boy named Max. Max loved his red ball more than anything. One day, the ball",
    "Anna was sad because she lost her favorite toy. She looked everywhere until she",
    "The old dog Rex was sleeping under the tree when suddenly he heard a",
    "Mia and her best friend Sam wanted to build a sandcastle at the beach, so they",
    "It was a cold winter morning. Sara put on her warm coat and went outside to",
    "Ben found a shiny key on the ground. He wondered what it could",
    "The little bird could not fly yet. Every day, its mother taught it to",
    "Grandma was baking cookies in the kitchen. The smell made everyone",
    "Two friends, Leo and Mia, decided to explore the old forest behind their house. They",
    "The kitten was stuck in a tall tree and could not get down. A boy named Sam",
    "Every morning, the farmer fed his animals. Today, the little pig ran away, so the farmer",
    "Emma got a new bicycle for her birthday. She was so excited that she",
    "There was a small boat on the lake. Two children, Amy and Jack, wanted to",
    "The wind blew hard and knocked over all the flower pots. Mom asked Tim to",
    "A little mouse lived in a hole in the wall. One day, it found a piece of",
    "Sophie loved to draw pictures of animals. Today she wanted to draw a",
    "The children were playing hide and seek in the yard. It was Ollie's turn to",
    "It began to rain suddenly during the picnic. Everyone ran to",
    "The puppy chewed on Dad's shoe and made a big mess. Dad was",
    "In the middle of the night, Lily heard a strange noise outside her window. She",
    "The three friends built a treehouse together over the summer. Inside, they kept",
    "A butterfly landed on Ella's hand while she was sitting in the garden. She",
    "Jake dropped his ice cream on the ground and started to cry. His big brother",
    "The old clock in the hallway stopped working. Grandpa tried to",
    "Zoe wanted to help her mom cook dinner, so she",
    "The little duckling got separated from its family at the pond. A kind girl named Ruby",
    "On the first day of school, Noah was very nervous. His teacher smiled and said,",
    "The two brothers found an old chest buried in the backyard. Inside, there was",
    "Every night before bed, Daddy read Mia a story about a",
    "The snow fell all night, and in the morning the whole town was covered in white. The children",
    "A little fish named Finn wanted to see what was beyond the coral reef, so he",
    "Mom told Tom not to touch the hot stove, but he was curious, so he",
    "The class went on a trip to the zoo. Everyone's favorite animal was the",
    "Lucas built a tower out of blocks, but his little sister knocked it down. He felt",
    "The stray cat outside the house looked hungry and cold. Emily decided to",
    "It was the last day of summer, and the friends wanted to do something special. They",
    "A strong wind carried Sam's kite far into the sky. He chased after it and",
    "The garden was full of colorful flowers. Grandma taught Lily how to",
    "One night, a little star fell from the sky into the backyard. The children found it and",
    "Ben's dog ran into the woods and did not come back. Ben and his dad went to",
    "The two sisters wanted to make a lemonade stand. First, they",
    "A loud thunder scared the little rabbit, and it hid under the",
    "Every summer, the family went camping by the lake. This year, they brought a new",
    "The little boy could not find his teddy bear anywhere. He asked his mom, and she",
]

assert len(EVAL_PROMPTS) == 50, f"expected 50 prompts, got {len(EVAL_PROMPTS)}"

_GRADING_PROMPT_TEMPLATE = """The following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion:

{story_beginning} *** {completion}

Please provide your general assessment about the part written by the student (the one after the *** symbol). Is it gramatically correct? Is it consistent with the beginning of the story? Now, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

Respond in exactly this format, with no other text:
Grammar: X/10, Creativity: X/10, Consistency: X/10, Age group: X"""

_SCORE_RE = re.compile(
    r"Grammar:\s*(\d+)\s*/\s*10.*?Creativity:\s*(\d+)\s*/\s*10.*?Consistency:\s*(\d+)\s*/\s*10.*?Age group:\s*([A-F])",
    re.IGNORECASE | re.DOTALL
)


@dataclass
class CompletionGrade:
    grammar: float
    creativity: float
    consistency: float
    age_group: str


@dataclass
class PromptResult:
    prompt: str
    completions: List[str]
    grades: List[CompletionGrade]

    def mean_scores(self) -> Dict[str, float]:
        if not self.grades:
            return {"grammar": float("nan"), "creativity": float("nan"), "consistency": float("nan")}
        return {
            "grammar": statistics.mean(g.grammar for g in self.grades),
            "creativity": statistics.mean(g.creativity for g in self.grades),
            "consistency": statistics.mean(g.consistency for g in self.grades),
        }


def generate_completions(
    generator: VeraceV1Generator,
    prompts: List[str] = EVAL_PROMPTS,
    completions_per_prompt: int = COMPLETIONS_PER_PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, List[str]]:
    """Generates `completions_per_prompt` completions per prompt at temperature=1.0,
    matching the paper's sampling protocol. Returns {prompt: [completion, ...]}."""
    results: Dict[str, List[str]] = {}
    for prompt in prompts:
        completions = []
        for _ in range(completions_per_prompt):
            completion = generator.generate(
                prompt_text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=GENERATION_TEMPERATURE,
                use_tree_search=False,  # paper's protocol is plain temperature sampling
            )
            completions.append(completion)
        results[prompt] = completions
    return results


def _parse_grade(response_text: str) -> Optional[CompletionGrade]:
    match = _SCORE_RE.search(response_text)
    if not match:
        return None
    grammar, creativity, consistency, age_group = match.groups()
    return CompletionGrade(
        grammar=float(grammar), creativity=float(creativity), consistency=float(consistency),
        age_group=age_group.upper()
    )


def grade_completion_with_gpt4(story_beginning: str, completion: str, model: str = "gpt-4") -> Optional[CompletionGrade]:
    """Calls GPT-4 with the paper's exact grading template, parses the response into a
    CompletionGrade. Returns None if the response couldn't be parsed (caller should treat
    that prompt/completion pair as ungraded rather than silently zero-filling it)."""
    from openai import OpenAI
    client = OpenAI()
    grading_prompt = _GRADING_PROMPT_TEMPLATE.format(story_beginning=story_beginning, completion=completion)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": grading_prompt}],
        temperature=0.0,
    )
    return _parse_grade(response.choices[0].message.content)


def run_evaluation(
    generator: VeraceV1Generator,
    prompts: List[str] = EVAL_PROMPTS,
    completions_per_prompt: int = COMPLETIONS_PER_PROMPT,
    grading_model: str = "gpt-4",
    save_path: Optional[str] = None,
) -> List[PromptResult]:
    """
    Full pipeline: generate completions_per_prompt completions per prompt, grade every
    one with GPT-4 using the paper's exact rubric, return per-prompt results (with
    per-completion grades so mean_scores() / overall aggregation can be computed).

    Raises RuntimeError immediately if OPENAI_API_KEY isn't set -- grading is the whole
    point of this function, so silently producing ungraded output would be worse than
    failing loudly before spending any generation compute.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "run_evaluation() requires OPENAI_API_KEY (GPT-4 grading, matching the "
            "TinyStories paper's methodology) -- set it before running, e.g. "
            "`export OPENAI_API_KEY=...`. Not silently skipping grading."
        )

    prompt_completions = generate_completions(generator, prompts, completions_per_prompt)

    results: List[PromptResult] = []
    for prompt, completions in prompt_completions.items():
        grades = []
        for completion in completions:
            grade = grade_completion_with_gpt4(prompt, completion, model=grading_model)
            if grade is not None:
                grades.append(grade)
        results.append(PromptResult(prompt=prompt, completions=completions, grades=grades))

    if save_path:
        with open(save_path, "w") as f:
            json.dump(
                [
                    {
                        "prompt": r.prompt,
                        "completions": r.completions,
                        "grades": [g.__dict__ for g in r.grades],
                        "mean_scores": r.mean_scores(),
                    }
                    for r in results
                ],
                f, indent=2
            )

    return results


def summarize_results(results: List[PromptResult]) -> Dict[str, float]:
    """Overall averages across every prompt's mean_scores() -- the paper's final
    reported numbers are exactly this: per-prompt averages over 10 completions, then
    averaged again across all prompts."""
    per_prompt = [r.mean_scores() for r in results if r.grades]
    if not per_prompt:
        return {"grammar": float("nan"), "creativity": float("nan"), "consistency": float("nan"), "prompts_graded": 0}
    return {
        "grammar": statistics.mean(s["grammar"] for s in per_prompt),
        "creativity": statistics.mean(s["creativity"] for s in per_prompt),
        "consistency": statistics.mean(s["consistency"] for s in per_prompt),
        "prompts_graded": len(per_prompt),
        "prompts_total": len(results),
    }
