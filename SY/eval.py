import time
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.neuron import NeuronModelForCausalLM

# =========================================================
# 1️⃣ 경로 설정
# =========================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL_PATH = "/home/ubuntu/environment/ml/SY/qwen/compiled_model"
FINETUNED_TOKENIZER_PATH = "/home/ubuntu/environment/ml/SY/qwen/merged_model"

EVAL_SIZE = 20  # 평가 샘플 수

# =========================================================
# 2️⃣ 모델 로드
# =========================================================
print("Loading baseline model (transformers) ...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("Loading fine-tuned Neuron compiled model ...")
ft_model = NeuronModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH)
ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_TOKENIZER_PATH)

# =========================================================
# 3️⃣ 평가 데이터셋 로딩
# =========================================================
print("Loading evaluation subset ...")
dataset = load_dataset("lextale/FirstAidInstructionsDataset", split=f"Superdataset[:{EVAL_SIZE}]")

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
records = []

# =========================================================
# 4️⃣ 공통 함수 정의
# =========================================================
def generate(model, tokenizer, prompt, max_new_tokens=200):
    """텍스트 생성 및 지연 시간 측정"""
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    latency = time.perf_counter() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, latency

# =========================================================
# 5️⃣ 평가 루프
# =========================================================
print(f"Evaluating {EVAL_SIZE} samples ...")
for sample in tqdm(dataset):
    q, gold = sample["question"], sample["answer"]

    prompt = (
        "<|im_start|>system\n"
        "You are a calm and accurate first-aid assistant. "
        "Provide safe, step-by-step emergency instructions.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

    # ---- baseline ----
    base_pred, base_latency = generate(base_model, base_tokenizer, prompt)
    rouge_base = scorer.score(gold, base_pred)["rougeL"].fmeasure

    # ---- fine-tuned ----
    ft_pred, ft_latency = generate(ft_model, ft_tokenizer, prompt)
    rouge_ft = scorer.score(gold, ft_pred)["rougeL"].fmeasure

    records.append({
        "question": q,
        "gold": gold,
        "rouge_base": rouge_base,
        "rouge_ft": rouge_ft,
        "latency_base": base_latency,
        "latency_ft": ft_latency,
        "speedup_ratio": base_latency / ft_latency if ft_latency > 0 else None
    })

df = pd.DataFrame(records)
df.to_csv("firstaid_qwen3_comparison.csv", index=False)
print("✅ Saved results → firstaid_qwen3_comparison.csv")

# =========================================================
# 6️⃣ 통계 요약
# =========================================================
summary = {
    "Avg ROUGE-L (Base)": df["rouge_base"].mean(),
    "Avg ROUGE-L (Finetuned)": df["rouge_ft"].mean(),
    "Avg Latency (Base)": df["latency_base"].mean(),
    "Avg Latency (Finetuned)": df["latency_ft"].mean(),
    "Avg Speedup": df["speedup_ratio"].mean(),
}
print("\n===== SUMMARY =====")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")

# =========================================================
# 7️⃣ 시각화: 품질–속도 트레이드오프
# =========================================================
plt.figure(figsize=(7,5))
plt.scatter(df["latency_base"], df["rouge_base"], label="Base Qwen3", alpha=0.6)
plt.scatter(df["latency_ft"], df["rouge_ft"], label="Fine-tuned Qwen3", alpha=0.6, color="orange")
plt.xlabel("Latency (s)")
plt.ylabel("ROUGE-L")
plt.title("Latency–Quality Comparison: Base vs Fine-tuned Qwen3")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_quality_comparison.png", dpi=300)
print("📈 Saved plot → latency_quality_comparison.png")

# =========================================================
# 8️⃣ 보조 시각화: ROUGE-L 분포
# =========================================================
plt.figure(figsize=(6,4))
plt.hist(df["rouge_base"], bins=20, alpha=0.6, label="Base")
plt.hist(df["rouge_ft"], bins=20, alpha=0.6, label="Fine-tuned")
plt.xlabel("ROUGE-L")
plt.ylabel("Count")
plt.title("ROUGE-L Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("rouge_distribution.png", dpi=300)
print("📊 Saved plot → rouge_distribution.png")
