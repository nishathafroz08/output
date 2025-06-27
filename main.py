from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import os

from app.generator_refactored import generate_retrieval_questions as generate_retrieval_handler
from app.Pretrainedmodel import (
    load_t5_model_and_tokenizer,
    load_qa_pipeline_model,
    load_json_dataset,
    filter_dataset,
    generate_questions_batch,
    get_semantically_relevant_distractors_batch,
    assemble_mcq,
    set_seed,
    INPUT_PATH,
    MODEL_PATH,
    USE_QA_PREDICTION
)
from sentence_transformers import SentenceTransformer
import random
import time
from concurrent.futures import ThreadPoolExecutor

# ------------------- Load Config -------------------
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("Missing config.json. App cannot start.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

GENERATOR_MODE = config.get("generator_mode")

# ------------------- FastAPI Init -------------------
app = FastAPI(title="Smart MCQ Generator API")

# ------------------- Pydantic Models -------------------
class QuestionItem(BaseModel):
    type: str
    question: str
    answer: str
    difficulty: str
    topic: str
    options: Optional[List[str]] = None

class MCQRequest(BaseModel):
    goal: str
    difficulty: str
    num_questions: int

class QuizResponse(BaseModel):
    goal: str
    difficulty: str
    questions: List[QuestionItem]

# ------------------- Retrieval Endpoint -------------------
@app.post("/generate/retrieval", response_model=QuizResponse, response_model_exclude_none=True)
def generate_retrieval_questions(request: MCQRequest):
    try:
        result = generate_retrieval_handler(request)
        if not result or not isinstance(result, dict) or "questions" not in result:
            raise HTTPException(status_code=500, detail="Retrieval question generation failed")

        questions = result.get("questions", [])
        cleaned_questions = []
        for idx, q in enumerate(questions):
            required_keys = ["type", "question", "answer", "difficulty", "topic"]
            if not all(k in q and q[k] for k in required_keys):
                raise HTTPException(status_code=500, detail=f"Invalid question at index {idx}")

            if q["type"].lower() == "mcq":
                if "options" not in q or not isinstance(q["options"], list) or not q["options"]:
                    raise HTTPException(status_code=500, detail=f"MCQ at index {idx} missing options")
            elif q["type"].lower() == "short_answer":
                q.pop("options", None)
            else:
                raise HTTPException(status_code=500, detail=f"Unsupported type '{q['type']}' at index {idx}")

            cleaned_questions.append(q)

        return QuizResponse(
            goal=result.get("goal"),
            difficulty=result.get("difficulty"),
            questions=cleaned_questions
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- Model Endpoint -------------------
@app.post("/generate/model", response_model=QuizResponse, response_model_exclude_none=True)
def generate_model_questions(request: MCQRequest):
    try:
        SUPPORTED_GOALS = config.get("model_supported_goals", [])
        SUPPORTED_DIFFICULTIES = config.get("supported_difficulties", [])
        DEFAULT_NUM_QUESTIONS = config.get("default_num_questions", 5)
        MAX_NUM_QUESTIONS = config.get("max_questions", 10)
        SBERT_PATH = config["sentence_transformer_path"]

        goal = request.goal
        difficulty = request.difficulty
        num_questions = min(request.num_questions, MAX_NUM_QUESTIONS)

        if goal not in SUPPORTED_GOALS or difficulty not in SUPPORTED_DIFFICULTIES:
            raise HTTPException(status_code=400, detail="Unsupported goal or difficulty")

        tokenizer, model, device = load_t5_model_and_tokenizer(MODEL_PATH)
        qa_pipeline = load_qa_pipeline_model() if USE_QA_PREDICTION else None
        dataset = load_json_dataset(INPUT_PATH)
        sbert_model = SentenceTransformer(SBERT_PATH)

        filtered = filter_dataset(dataset, goal, difficulty)
        if not filtered:
            raise HTTPException(status_code=404, detail="No data found for the filters")

        seen = set()
        unique_items = []
        for item in filtered:
            key = (item["context"].strip().lower(), item["correct_answer"].strip().lower())
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        if not unique_items:
            raise HTTPException(status_code=404, detail="No unique pairs found")

        selected = random.sample(unique_items, min(num_questions, len(unique_items)))
        contexts = [item["context"].strip() for item in selected]
        answers = [item["correct_answer"].strip() for item in selected]
        distractor_lists = [
            [d for d in item.get("distractors", []) if d.strip().lower() != answers[i].lower()]
            for i, item in enumerate(selected)
        ]

        set_seed(int(time.time()) % 100000)

        questions = generate_questions_batch(contexts, answers, model, tokenizer, device, goal)
        filtered_distractors = get_semantically_relevant_distractors_batch(
            answers, distractor_lists, sbert_model
        )

        with ThreadPoolExecutor() as executor:
            mcqs = list(executor.map(
                assemble_mcq,
                range(len(selected)),
                selected,
                contexts,
                questions,
                answers,
                filtered_distractors,
                [qa_pipeline] * len(selected),
                [USE_QA_PREDICTION] * len(selected)
            ))

        return QuizResponse(
            goal=goal,
            difficulty=difficulty,
            questions=mcqs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- Dynamic /generate Dispatcher -------------------
@app.post("/generate", response_model=QuizResponse, response_model_exclude_none=True)
def generate_dynamic(request: MCQRequest):
    if GENERATOR_MODE == "retrieval":
        return generate_retrieval_questions(request)
    elif GENERATOR_MODE == "model":
        return generate_model_questions(request)
    else:
        raise HTTPException(status_code=400, detail="Invalid generator_mode in config.json")

@app.get("/")
def root():
    return {"message": "Welcome to the Smart MCQ Generator API"}

# ------------------- Run the app -------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
