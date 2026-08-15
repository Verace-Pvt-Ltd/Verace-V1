"""
TinyStories paper's GPT-4 evaluation methodology (Eldan & Li, arXiv:2305.07759), adapted
for Verace V1: for each of the paper's 44 manually-constructed story-beginning prompts,
generate 10 completions at temperature=1, ask GPT-4 to grade each completion on Grammar,
Creativity, and Consistency (0-10 each) plus a guessed writer age-group, and average the
scores.

EVAL_PROMPTS below is the paper's actual prompt set, not an approximation: it's
"Evaluation prompts.yaml" from the paper authors' own dataset repo
(https://huggingface.co/datasets/roneneldan/TinyStories, file "Evaluation prompts.yaml"),
which is not quoted or listed in the paper's PDF/appendix itself. Confirmed authentic by
spot-checking against the paper's own worked examples: prompt #31 (0-indexed 30) is the
"Lucy and the ladder" story completed in Figure 6, #33 is the talking-pumpkin story in
Figure 7, and #44 is the black-cat story in Figure 8 -- all three match the paper's
figures verbatim. This means absolute scores computed here ARE directly comparable to the
paper's published table (Figure 4) and per-example grades (Figures 6-8), not just
methodologically parallel.

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

# The paper's actual 44-prompt eval set (see module docstring for provenance/verification).
EVAL_PROMPTS: List[str] = [
    "Once upon a time, there lived a bunny in a field. Her name was Lucy. Lucy loved to have feasts and parties with her bunny friends. One day, when Lucy was about to leave for a feast at a friend's house, she realized she's starting to feel sick. She was so weak she could",
    "One day a girl walked into the living room and noticed something very strange. There was a huge cabinet standing in the corner. It looked very old and heavy. She walked over and tried to open it, when suddenly",
    "Once upon a time, there lived a hamster in the forest. Every day, he would walked around the forest and looking for adventures. One day, he heard someone calling out from behind the bushes.\n\nThe hamster listened carefully. He realised that it was a small mouse calling out for help. It got stuck under a heavy log and couldn't get out. The hamster immediately realized that",
    "Jack asked his mom if he could ride the bike all the way to his grandmother's house. She agreed, but she said that he shouldn't ride too fast because it was raining and the path was wet.\n\nHe started riding and realized he was hungry, so he decided that he should get to his grandmother's house as fast as possible. But then he remembered his mum's words \"",
    "Alice was bored and wanted to find some adventures. She walked up to her friend Ben, who looked vey busy playing with his toys. Alice said, \"Why don't we",
    "Alice walked up to her friend Ben's house. She was planning to ask him to go to the park with her. When Ben opened the door, she asked him if he had any plans. He said, \"I'm sorry, Alice, but",
    "The day before Ben's birthday, Alice walked into the kitchen and saw Ben sitting there, looking Gloomy. She said, \"Ben, why are you",
    "Alice walked into the kitchen and saw Ben who was looking for something but looked frustrated. She said, \"Ben, why are you",
    "Once upon a time there was a curious boy who lived in a house with a big garden. Every day he explored the garden he found new surprises. But one day, it was raining so hard that his mother told him",
    "Once upon a time, there was tiger who liked to play the guitar. One day, a bunny heard the guitar from a distance and",
    "Alice wanted to play with her doll, but she couldn't remember where she had put it. She looked all around the house but couldn't find it, so she decided",
    "\"Ben, what do you have in your pocket?\", Alice asked. \"Oh, nothing.\", Ben replied. But Alice saw that there was definitely something in Ben's pocket, and she was very curious what it was, so she",
    "One day, a bird was flying high over the sea. At some point the bird noticed small boat with a boy sitting inside. The boy looked lost so",
    "Jimmy was on his way to school with his father when he noticed a tree with strange looking fruits on it. He asked his father, \"Can I try",
    "Once upon a time, there lived two best friends called Jack and Joe. They were bored, they wanted someone else to talk to or play with. Jack had an idea. He said to Joe, “Let’s go and find",
    "Jack and Sally were running through the woods together when they stopped suddenly. They heard a strange sound coming from around the corner.\n\n“What's that?” asked Jack.\n\nSally put her finger to her lips and said, “Shhhh. Let's go find out.”\n\nJack and Sally followed the",
    "Once upon a time there was a little bear named Diva. She was hungry, and wanted to bake a cake, but she didn't have any sugar at home, so she decided to go ask around. She started walking and met a squirrel. She asked the squirrel, \"Would you happen",
    "Anna was a popular girl, who walked to school every day carrying a very heavy backpack. It was so heavy that she could hardly walk.\n\nOne day, Anna's friends asked her why her backpack was so heavy. She said that",
    "Once there was a young girl named Anna. She was three years old and usually very happy. Today she was sad because she couldn't play in the park. She had asked her parents to take her but they wouldn't listen. Anna insisted that ",
    "One day, Grandma was in the kitchen getting ready to prepare. She had the ingredients for a tasty dish.\n\nGrandma said to her granddaughter, \"Are you going to help me make this meal?\" Her granddaughter, Jenny, replied, \"Yes, I will help you!\"\n\nGrandma and Jenny began to prepare the ingredients. Jenny was clumsy. She dropped",
    "Once upon a time there was a boy named Max. He was at the market with his mother, where he saw two things he wanted very much: a toy car and a jumping rope.\n\nHe knew that his mother would not let him buy both of these things, and that she would probably tell him that he has to choose one. He said to her, \"I want both",
    "Once upon a time there was a small brown mouse who lived in a big tree. The mouse wanted to be better friends with the birds that flew around the tree, but they were always too afraid to let him get close. \n\nOne day, the mouse spoke to the birds bravely, \"Please don't be",
    "Once upon a time there was a mummy, a daddy and a little girl. The family lived in a house and the little girl liked to play in the living room. Her favourite spot was a big, comfy armchair. \n\nOne day, the little girl was playing in the armchair when mummy and daddy said it was time to go. She didn't want to leave, she wanted to stay in her",
    "Jack was a chubby little boy. He liked to pick flowers in the garden. One day, he went out to the garden to pick some flowers but his Mom said he had to take his medicine first. He didn't like the medicine's taste.\n\n\"Do I have to take it?\" Jack asked.\n\nHis Mom nodded. \"Yes,",
    "Once upon a time, there lived a small dog named Spike. He was an adventurous pup who loved to explore the world. One day, while he was wandering around, he stumbled upon a big tree. He saw something shining at the top. He wanted to climb it to see what it was.\n\nSpike placed his paws on the trunk and started to climb, but the trunk was too",
    "Once there was a mother and daughter walking in the park. They noticed a bird perched on a branch, singing its song. The daughter shouted with delight, “Look Mommy! A bird!”\n\nThe mother smiled and said, “Yes! See how colorful it is?”\n\nThe daughter then asked, “Can I share my snack with it?”\n\nThe mother replied, “No, sweetie. That's a bad idea because",
    "Once upon a time there were two friends, Max and Alex. One day Max invited Alex to his house for lunch. Alex was excited to accept, and so they went.\n\nWhen they arrived, Max told Alex that he had made soup for lunch. Alex was excited and thanked Max for his kind invitation. Max smiled and said it was easy to make.\n\nLater, it was time to eat the soup. Max took the first spoonful and tasted it. He realized something must be wrong with the soup, because it tasted very weird. He said to Alex, \"This soup",
    "Once upon a time there was a little girl called Emma. She was only three years old. One day, her aunt said to her, “Emma, would you like to go out with me to pick up some flowers?”\n\nEmma was very excited. She said,",
    "Once upon a time there was a little girl named Lucy. She loved to explore the world around her, especially when it was bright and sunny outside. \n\nOne day, while exploring the nearby park, Lucy came across an old and long rock wall. She was curious and wanted to get to the top, so she began to look around for something she could use to climb it. Suddenly, she spotted a",
    "Once upon a time there was a small girl named Lucy. She was only three years old and was very adventurous. She loved to explore the world around her, especially when it was bright and sunny outside. \n\nOne day, while exploring the nearby park, Lucy came across a ladder leaning on a wall. She was curious to see what's on top, so she climbed the ladder, but when she reached the top, the ladder fell and she was stuck.\n\nA nearby park ranger noticed her and shouted out, \"",
    "Once upon a time, in a far away land, two best friends, Bob and Jill, were playing hide and seek. Jill hid below a large wooden table. She was very quiet so that",
    "Once upon a time there was a pumpkin. It was a very special pumpkin, it could speak. It was sad because it couldn't move. Every day, it would say",
    "Once upon a time there were two friends, Sally and Joe who liked to go to each other's house. One day, Joe wanted to visit Sally so he walked up to her house. He knocked on the door, but",
    "Mum was getting ready for bed, so she told her son, \"It's time for a bath\". He nodded and ran off to the bathroom. He watched as Mum turned on the bathtub, the water pouring out of the tap and filling the tub.\n\nMum asked, \"What do you want in your bath?\" He thought for a moment and said",
    "Once upon a time, there was a jolly boy named Dan. He was a youth of three years old. He liked to play outside in the bright sunshine.\n\nOne day, Dan was playing in his garden when he saw something very exciting.\n\n“Look, Mama!” shouted Dan, “I see something moving!”\n\nMama peered around the corner to see what Dan was pointing at. It was a",
    "Once upon a time there was a little boy named Tom. Tom loved playing outside, but he always had to be careful not to make too much noise.\n\nOne day, Tom went out to explore the garden. Soon he found an unusual fruit. It looked like a prune! Tom picked up the prune and looked at it closely. It smelled sweet.\n\nTom wanted to find out how the prune tasted, so he",
    "Once there were two friends who were walking together in the park. One had a cane and the other didn't. The one without the cane wanted to borrow the cane from the other. But the other friend said",
    "One day, Timmy had a goal. He wanted to fit into his favorite toy car. He was very excited, but grumpy too. He tried to fit into the car but he was",
    "Once upon a time, there was a little girl. Her name was Julia. Every day, Julia helped her mama do the laundry. It was her favourite job. She loved to put the",
    "It was a sunny day and Ben wanted to go outside and play. He grabbed his toy car and ran to the garden. Suddenly, something pinched his hand!\n\nBen looked around but there was nothing there. He was scared so he held onto the hoop tightly.\n\nThen he heard a voice. \"Don't be scared, little one. It's just me!\"\n\nBen looked up and saw a big, scary spider. He was scared!\n\nThe spider smiled. \"Don't be scared, I just wanted to",
    "Once upon a time there was a little girl. She liked to go out and explore the world. One day she decided to go for a walk in the forest.\n\nAs she was walking, she spotted a mysterious house. She stepped closer to get a better look. Suddenly, a witch appeared from the door, and the little girl was very scared. \n\n\"What do you want?\" asked the witch.\n\nThe little girl was brave and said, \"I want to explore this house and see",
    "Once there was a boy named Sam. He was three years old and very naughty. One day, his mom told him not to touch her phone but Sam did not listen. He picked up the phone and started playing with it.\n\nMom saw him and said, \"be careful not to",
    "Once upon a time, there lived a black cat. The cat belonged to a little girl called Katie. Every day, Katie would take her cat for a walk in the park. \n\nOne day, as Katie and her cat were walking around, they saw a mean looking man. He said he wanted to take the cat, to which she replied \"This cat belongs",
    "Once upon a time, there was a little boy who was always naughty. His mom was always telling him to be good, but he kept disobeying her rules and ignoring her warnings. \n\nOne day, he was so naughty that his mom decided to punish him. She told him that he had to",
]

assert len(EVAL_PROMPTS) == 44, f"expected 44 prompts, got {len(EVAL_PROMPTS)}"

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
