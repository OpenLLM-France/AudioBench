"""Flow-Judge v0.1 as evaluation judge (vLLM backend)."""

import torch
from flow_judge import Vllm, FlowJudge, EvalInput
from flow_judge.metrics import RESPONSE_CORRECTNESS_5POINT, RESPONSE_CORRECTNESS_BINARY


def _run_flow_judge(metric, input_data):
    """Shared logic for both 5-point and binary variants."""
    model = Vllm(gpu_memory_utilization=0.2)
    judge = FlowJudge(metric=metric, model=model, output_dir=None)

    questions, references, predictions = input_data

    eval_inputs = [
        EvalInput(
            inputs=[{"query": q}, {"reference_answer": r}],
            output={"response": p},
        )
        for q, r, p in zip(questions, references, predictions)
    ]

    results = judge.batch_evaluate(eval_inputs, save_results=False)

    all_details = []
    for q, r, p, result in zip(questions, references, predictions, results):
        all_details.append({
            "question": q,
            "reference": r,
            "model_prediction": p,
            "judge_response": result.feedback,
            "rate_score": result.score if result.score is not None else 0,
            "success": 1 if result.score is not None and result.score >= 0 else 0,
        })

    del model
    torch.cuda.empty_cache()

    return all_details


def flow_judge_as_judge(model_path, input_data):
    """5-point scoring (1-5). Returns (results_dict, all_details)."""
    all_details = _run_flow_judge(RESPONSE_CORRECTNESS_5POINT, input_data)

    all_scores = [d["rate_score"] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 20  # scale 1-5 -> 0-100
    success_rate = sum(d["success"] for d in all_details) / len(all_details)

    return {"judge_score": avg_score, "success_rate": success_rate}, all_details


def flow_judge_as_judge_binary(model_path, input_data):
    """Binary scoring (0-1). Returns (results_dict, all_details)."""
    all_details = _run_flow_judge(RESPONSE_CORRECTNESS_BINARY, input_data)

    all_scores = [d["rate_score"] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 100  # scale 0-1 -> 0-100
    success_rate = sum(d["success"] for d in all_details) / len(all_details)

    return {"judge_score": avg_score, "success_rate": success_rate}, all_details
