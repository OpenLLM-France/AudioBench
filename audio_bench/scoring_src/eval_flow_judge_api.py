"""Flow-Judge v0.1 as evaluation judge via external vLLM API."""

import os
import re

from tqdm import tqdm
from openai import OpenAI
from multiprocessing import Pool


# Exact replica of the flow-judge package prompt template
# (from flow_judge.utils.prompt_formatter.USER_PROMPT_TEMPLATE)
USER_PROMPT_TEMPLATE = """\
# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large \
language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria \
and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{INPUTS}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{OUTPUT}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
{EVALUATION_CRITERIA}
</evaluation_criteria>

<scoring_rubric>
{RUBRIC}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""


# 5-point rubric (from flow_judge.metrics.presets.RESPONSE_CORRECTNESS_5POINT)
FIVE_POINT_CRITERIA = (
    "Compare the system's response to the provided reference answer and rate how well they match "
    "in accuracy and completeness to answer the query."
)
FIVE_POINT_RUBRIC = (
    "- Score 1: The response is completely incorrect or irrelevant to the query, "
    "with no overlap in information with the reference answer.\n"
    "- Score 2: The response contains some correct information relevant to the query "
    "but is substantially incomplete or inaccurate compared to the reference answer.\n"
    "- Score 3: The response answers the query with reasonable accuracy but is missing "
    "key details or has minor inaccuracies compared to the reference.\n"
    "- Score 4: The response accurately answers the query and is nearly complete, "
    "only leaving out non-essential details compared to the reference.\n"
    "- Score 5: The response perfectly matches the accuracy and level of detail of "
    "the reference answer, containing all key information to comprehensively answer the query."
)

# Binary rubric (from flow_judge.metrics.presets.RESPONSE_CORRECTNESS_BINARY)
BINARY_CRITERIA = (
    "Does the generated response accurately match the provided reference answer "
    "for the given query?"
)
BINARY_RUBRIC = (
    "- Score 0: The generated response does not match the reference answer. It either contains "
    "inaccurate information, is missing key details from the reference, includes extra information "
    "not in the reference, or fails to convey the same meaning as the reference answer.\n"
    "- Score 1: The generated response matches the reference answer exactly or contains all the "
    "key information from the reference with no inaccuracies, extra details, or missing details. "
    "The meaning conveyed by the generated response is equivalent to the reference."
)


def _build_prompt(question, reference, prediction, criteria, rubric, task_context=""):
    """Build the evaluation prompt matching the flow-judge package format."""
    inputs_str = f"<query>\n{question}\n</query>\n<reference_answer>\n{reference}\n</reference_answer>"
    if task_context:
        inputs_str += f"\n<task_context>\n{task_context}\n</task_context>"
    output_str = f"<response>\n{prediction}\n</response>"
    return USER_PROMPT_TEMPLATE.format(
        INPUTS=inputs_str,
        OUTPUT=output_str,
        EVALUATION_CRITERIA=criteria,
        RUBRIC=rubric,
    )


def _parse_response(output):
    """Parse score and feedback from flow-judge response using XML tags."""
    feedback_pattern = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.DOTALL)
    score_pattern = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)

    feedback_match = feedback_pattern.search(output)
    score_match = score_pattern.search(output)

    if feedback_match and score_match:
        return feedback_match.group(1).strip(), int(score_match.group(1).strip()), 1

    # Fallback: try to extract last number from response
    numbers = re.findall(r"\d+", output)
    if numbers:
        return output, int(numbers[-1]), 1

    return output, 0, 0


def _evaluate_one_sample(args):
    """Evaluate a single sample via the external vLLM API."""
    question, reference, prediction, criteria, rubric, task_context = args

    prompt = _build_prompt(question, reference, prediction, criteria, rubric, task_context)

    messages = [{"role": "user", "content": prompt}]

    port = os.environ.get('MY_VLLM_PORT_JUDGE', 5001)
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )

    models = client.models.list()
    model = models.data[0].id

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
        )
        output = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in completion: {e}")
        output = "empty"

    feedback, rate_score, success = _parse_response(output)

    return {
        'question': question,
        'reference': reference,
        'model_prediction': prediction,
        'judge_response': output,
        'rate_score': rate_score,
        'success': success,
    }


def flow_judge_api_as_judge(model_path, input_data, task_type=None, desc=None):
    """5-point scoring (1-5). Returns (results_dict, all_details)."""
    from audio_bench.scoring_src.metrics import get_task_evaluation_context
    questions, references, predictions = input_data
    task_context = get_task_evaluation_context(task_type)

    num_processes = min(8, len(questions))
    bar_desc = f"{desc} | Flow-Judge API (5-point)" if desc else "Flow-Judge API (5-point)"

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(
                    _evaluate_one_sample,
                    zip(
                        questions,
                        references,
                        predictions,
                        [FIVE_POINT_CRITERIA] * len(questions),
                        [FIVE_POINT_RUBRIC] * len(questions),
                        [task_context] * len(questions),
                    ),
                ),
                total=len(questions),
                desc=bar_desc,
            )
        )

    all_scores = [d['rate_score'] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 20  # scale 1-5 -> 0-100
    success_rate = sum(d['success'] for d in all_details) / len(all_details)

    return {'judge_score': avg_score, 'success_rate': success_rate}, all_details


def flow_judge_api_as_judge_binary(model_path, input_data, task_type=None, desc=None):
    """Binary scoring (0-1). Returns (results_dict, all_details)."""
    from audio_bench.scoring_src.metrics import get_task_evaluation_context
    questions, references, predictions = input_data
    task_context = get_task_evaluation_context(task_type)

    num_processes = min(8, len(questions))
    bar_desc = f"{desc} | Flow-Judge API (binary)" if desc else "Flow-Judge API (binary)"

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(
                    _evaluate_one_sample,
                    zip(
                        questions,
                        references,
                        predictions,
                        [BINARY_CRITERIA] * len(questions),
                        [BINARY_RUBRIC] * len(questions),
                        [task_context] * len(questions),
                    ),
                ),
                total=len(questions),
                desc=bar_desc,
            )
        )

    all_scores = [d['rate_score'] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 100  # scale 0-1 -> 0-100
    success_rate = sum(d['success'] for d in all_details) / len(all_details)

    return {'judge_score': avg_score, 'success_rate': success_rate}, all_details
