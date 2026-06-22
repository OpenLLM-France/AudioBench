"""Flow-Judge v0.1 as evaluation judge (vLLM backend)."""

import os
import logging

import torch
from flow_judge import Vllm, FlowJudge, EvalInput
from flow_judge.metrics import RESPONSE_CORRECTNESS_5POINT, RESPONSE_CORRECTNESS_BINARY

from audio_bench.scoring_src.metrics import get_task_evaluation_context


def _quiet_vllm_logs():
    """Silence vLLM's INFO chatter during judge scoring so the banner / progress
    stay readable. Sets the env var (read by the vLLM V1 engine subprocess at
    startup) and the in-process logger levels. Returns a callable that restores
    the previous state so inference-time vLLM logs elsewhere are unaffected."""
    prev_env = os.environ.get("VLLM_LOGGING_LEVEL")
    prev_levels = {name: logging.getLogger(name).level for name in ("vllm", "flow_judge")}

    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    for name in prev_levels:
        logging.getLogger(name).setLevel(logging.WARNING)

    def restore():
        if prev_env is None:
            os.environ.pop("VLLM_LOGGING_LEVEL", None)
        else:
            os.environ["VLLM_LOGGING_LEVEL"] = prev_env
        for name, level in prev_levels.items():
            logging.getLogger(name).setLevel(level)

    return restore


def _run_flow_judge(metric, input_data, task_type=None):
    """Shared logic for both 5-point and binary variants."""
    questions, references, predictions = input_data

    restore_logs = _quiet_vllm_logs()
    try:
        model = Vllm(quantized=True, gpu_memory_utilization=0.3, max_model_len=6200, max_num_seqs=50)
        judge = FlowJudge(metric=metric, model=model, output_dir=None)

        task_context = get_task_evaluation_context(task_type)

        eval_inputs = [
            EvalInput(
                inputs=[{"query": f"{task_context}\n{q}" if task_context else q}, {"reference_answer": r}],
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
    finally:
        restore_logs()

    return all_details


def flow_judge_as_judge(model_path, input_data, task_type=None):
    """5-point scoring (1-5). Returns (results_dict, all_details)."""
    all_details = _run_flow_judge(RESPONSE_CORRECTNESS_5POINT, input_data, task_type=task_type)

    all_scores = [d["rate_score"] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 20  # scale 1-5 -> 0-100
    success_rate = sum(d["success"] for d in all_details) / len(all_details)

    return {"judge_score": avg_score, "success_rate": success_rate}, all_details


def flow_judge_as_judge_binary(model_path, input_data, task_type=None):
    """Binary scoring (0-1). Returns (results_dict, all_details)."""
    all_details = _run_flow_judge(RESPONSE_CORRECTNESS_BINARY, input_data, task_type=task_type)

    all_scores = [d["rate_score"] for d in all_details]
    avg_score = sum(all_scores) / len(all_scores) * 100  # scale 0-1 -> 0-100
    success_rate = sum(d["success"] for d in all_details) / len(all_details)

    return {"judge_score": avg_score, "success_rate": success_rate}, all_details
