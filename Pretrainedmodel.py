import os
import json
import random
import logging
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import time

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline
from sentence_transformers import SentenceTransformer, util

CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("Missing config.json. App cannot start.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# Allowed config
SUPPORTED_GOALS = config["model_supported_goals"]
SUPPORTED_DIFFICULTIES = config["supported_difficulties"]
DEFAULT_NUM_QUESTIONS = config["default_num_questions"]
MAX_NUM_QUESTIONS = config["max_questions"]

INPUT_PATH = config["model_dataset"]
MODEL_PATH = config["t5-model_path"]
SBERT_PATH = config["sentence_transformer_path"]
SPACY_MODEL = config["spacy_model"]

USE_QA_PREDICTION = False
QG_PROMPT_TEMPLATE = "generate question: domain: {goal} context: {context} answer: {answer}"
SAVE_OUTPUT = True

# =============================
# 🔧 Setup
# =============================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_user_selection(options, prompt):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    choice = input("Enter option number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
        print("Invalid selection. Exiting.")
        return None
    return options[int(choice) - 1]

def load_t5_model_and_tokenizer(path: str) -> Tuple[T5TokenizerFast, T5ForConditionalGeneration, torch.device]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path '{path}' does not exist.")
    tokenizer = T5TokenizerFast.from_pretrained(path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"✅ Loaded T5 model on {device}")
    return tokenizer, model, device

def load_qa_pipeline_model():
    qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    logger.info("✅ Loaded QA pipeline")
    return qa

def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])

def filter_dataset(dataset: List[Dict[str, Any]], goal: str, difficulty: str) -> List[Dict[str, Any]]:
    g = goal.lower().strip()
    d_levels = [d.strip().lower() for d in difficulty.split(",")]
    return [item for item in dataset if item.get("goal", "").lower() == g and item.get("difficulty", "").lower() in d_levels]

def generate_questions_batch(contexts: List[str], answers: List[str], model, tokenizer, device, goal: str) -> List[str]:
    prompts = [
        QG_PROMPT_TEMPLATE.format(goal=goal, context=context[:512], answer=answer)
        for context, answer in zip(contexts, answers)
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    seen = set()
    return [q for q in decoded if not (q.lower().strip() in seen or seen.add(q.lower().strip()))]

def get_semantically_relevant_distractors_batch(
    answers: List[str],
    distractor_lists: List[List[str]],
    sbert: SentenceTransformer,
    top_k: int = 3
) -> List[List[str]]:
    answer_embs = sbert.encode(answers, convert_to_tensor=True)
    results = []
    for i, distractors in enumerate(distractor_lists):
        if not distractors:
            results.append([])
            continue
        d_embs = sbert.encode(distractors, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(answer_embs[i], d_embs)[0]
        scored = [(distractors[j], float(sims[j])) for j in range(len(distractors)) if float(sims[j]) < 0.95]
        top_distractors = [d for d, _ in sorted(scored, key=lambda x: -x[1])[:top_k]]
        results.append(top_distractors)
    return results

def predict_answer(question: str, context: str, qa_pipeline) -> Tuple[str, float]:
    try:
        result = qa_pipeline(question=question, context=context)
        return result.get("answer", ""), result.get("score", 0.0)
    except Exception as e:
        logger.warning(f"QA prediction error: {e}")
        return "", 0.0

def assemble_mcq(
    idx: int,
    item: Dict[str, Any],
    context: str,
    question: str,
    answer: str,
    distractors: List[str],
    qa_pipeline=None,
    use_qa=False
) -> Dict[str, Any]:
    if use_qa and qa_pipeline:
        predicted, score = predict_answer(question, context, qa_pipeline)
        if predicted.strip():
            answer = predicted.strip()
    options = list(dict.fromkeys(distractors + [answer]))
    random.shuffle(options)
    return {
        "type": "mcq",
        "question": question,
        "options": options,
        "answer": answer,
        "difficulty": item.get("difficulty", ""),
        "topic": item.get("topic", "")
    }

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"✅ Seed set to {seed}")

def main():
    goal = get_user_selection(SUPPORTED_GOALS, "Select a Goal:")
    if not goal:
        return

    difficulty = get_user_selection(SUPPORTED_DIFFICULTIES, "\nSelect Difficulty Level:")
    if not difficulty:
        return

    num_q_input = input(f"\nEnter number of questions (default {DEFAULT_NUM_QUESTIONS}, max {MAX_NUM_QUESTIONS}): ").strip()
    if num_q_input == "":
        num_questions = DEFAULT_NUM_QUESTIONS
    elif num_q_input.isdigit():
        num_questions = min(int(num_q_input), MAX_NUM_QUESTIONS)
    else:
        print("Invalid number input. Using default.")
        num_questions = DEFAULT_NUM_QUESTIONS

    print(f"\n📘 Generating quiz for Goal: {goal}, Difficulty: {difficulty}, Questions: {num_questions}\n")

    set_seed(int(time.time()) % 100000)
    tokenizer, model, device = load_t5_model_and_tokenizer(MODEL_PATH)
    qa_pipeline = load_qa_pipeline_model() if USE_QA_PREDICTION else None
    sbert_model = SentenceTransformer(SBERT_PATH)

    dataset = load_json_dataset(INPUT_PATH)
    filtered = filter_dataset(dataset, goal, difficulty)
    if not filtered:
        logger.error("❌ No matching samples found")
        return

    unique_pairs = []
    seen_pairs = set()
    for item in filtered:
        ctx = item["context"].strip()
        ans = item["correct_answer"].strip()
        pair_key = (ctx.lower(), ans.lower())
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_pairs.append(item)

    if not unique_pairs:
        logger.error("❌ No unique context-answer pairs found.")
        return

    selected_items = random.sample(unique_pairs, min(num_questions, len(unique_pairs)))

    contexts = [item["context"].strip() for item in selected_items]
    answers = [item["correct_answer"].strip() for item in selected_items]
    distractor_lists = [
        [d for d in item.get("distractors", []) if d.strip().lower() != answers[i].lower()]
        for i, item in enumerate(selected_items)
    ]

    logger.info("🚀 Generating questions...")
    questions = generate_questions_batch(contexts, answers, model, tokenizer, device, goal)

    logger.info("🧠 Filtering distractors...")
    filtered_distractors = get_semantically_relevant_distractors_batch(
        answers, distractor_lists, sbert_model
    )

    logger.info("⚙️ Assembling MCQs...")
    with ThreadPoolExecutor() as executor:
        mcqs = list(executor.map(
            assemble_mcq,
            range(len(selected_items)),
            selected_items,
            contexts,
            questions,
            answers,
            filtered_distractors,
            [qa_pipeline] * len(selected_items),
            [USE_QA_PREDICTION] * len(selected_items)
        ))

    result = {"goal": goal, "difficulty": difficulty, "questions": mcqs}
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if SAVE_OUTPUT:
        os.makedirs("./output", exist_ok=True)
        out_file = f"./output/mcqs_{goal.replace(' ', '_')}_{difficulty}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"📁 Output saved to {out_file}")

if __name__ == "__main__":
    main()
