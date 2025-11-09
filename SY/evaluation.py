import time
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

# =========================================================
# 1️⃣  경로 설정
# =========================================================
# Neuron-compiled 모델 경로 (파인튜닝 후 optimum-cli export neuron 으로 생성된 폴더)
MODEL_PATH = "/home/ubuntu/environment/ml/SY/qwen/compiled_model"
TOKENIZER_PATH = "/home/ubuntu/environment/ml/SY/qwen/merged_model"

# 평가 샘플 수 (속도 테스트용)
EVAL_SIZE = 100

# =========================================================
# 2️⃣  모델 및 토크나이저 로드
# =========================================================
print("Loading Neuron compiled model ...")
model = NeuronModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# =========================================================
# 3️⃣  평가 데이터셋 로딩
# =========================================================
print("Loading FirstAidInstructionsDataset subset ...")
dataset = load_dataset(
    "lextale/FirstAidInstructionsDataset",
    split=f"Superdataset[:{EVAL_SIZE}]"
)

# =========================================================
# 4️⃣  평가 함수 정의
# =========================================================
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
results = []

def generate_answer(prompt, max_new_tokens=200):
    """Neuron inference + latency 측정"""
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    latency = time.perf_counter() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, latency

# =========================================================
# 5️⃣  평가 루프
# =========================================================
print(f"Running evaluation on {EVAL_SIZE} samples ...")
for sample in tqdm(dataset):
    q, gold = sample["question"], sample["answer"]
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful first-aid assistant. "
        "Provide safe and practical instructions.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

    pred, latency = generate_answer(prompt)
    rouge = scorer.score(gold, pred)["rougeL"].fmeasure
    results.append({
        "question": q,
        "gold": gold,
        "prediction": pred,
        "rougeL": rouge,
        "latency_s": latency,
    })

# =========================================================
# 6️⃣  결과 집계 및 출력
# =========================================================
df = pd.DataFrame(results)
summary = {
    "mean_ROUGE-L": df["rougeL"].mean(),
    "mean_latency_s": df["latency_s"].mean(),
    "median_latency_s": df["latency_s"].median(),
}
print("\n===== EVALUATION SUMMARY =====")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")

df.to_csv("firstaid_neuron_eval_results.csv", index=False)
print("Saved results → firstaid_neuron_eval_results.csv")
