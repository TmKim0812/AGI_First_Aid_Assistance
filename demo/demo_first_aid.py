import gradio as gr
import time
from vllm import LLM, SamplingParams

# =======================================================
# 1️⃣ Load local fine-tuned model (Neuron compiled)
# =======================================================
llm = LLM(
    model="/home/ubuntu/environment/ml/SY/qwen/compiled_model",
    max_num_seqs=1,
    max_model_len=2048,
    device="neuron",
    tensor_parallel_size=2,
)

sampling_params = SamplingParams(max_tokens=256, temperature=0.7)

# =======================================================
# 2️⃣ Core inference function
# =======================================================
def firstaid_response(user_input, chat_history):
    """
    Called each time user sends a message.
    Keeps conversation context and measures latency.
    """
    system_prompt = (
        "<|im_start|>system\n"
        "You are a calm, medically accurate first-aid assistant. "
        "Give concise, step-by-step instructions. "
        "If the situation may be dangerous, recommend calling emergency services.\n"
        "<|im_end|>\n"
    )

    # Reconstruct conversation context
    conversation = ""
    for turn in chat_history:
        conversation += f"<|im_start|>user\n{turn[0]}<|im_end|>\n"
        conversation += f"<|im_start|>assistant\n{turn[1]}<|im_end|>\n"
    conversation += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    prompt = system_prompt + conversation

    # Generate
    start_time = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.perf_counter() - start_time

    reply = outputs[0].outputs[0].text.strip()
    tokens = len(reply.split())
    speed = tokens / elapsed if elapsed > 0 else 0

    latency_info = f"⏱️ {elapsed:.2f}s | ⚡ {speed:.1f} tokens/s"
    full_reply = f"{reply}\n\n{latency_info}"

    chat_history.append((user_input, full_reply))
    return "", chat_history

# =======================================================
# 3️⃣ Gradio UI definition
# =======================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown(
        """
        # 🩺 **Real-time First Aid Assistant (SLM Prototype)**
        This demo showcases a fine-tuned **Qwen3-1.7B Small Language Model**  
        capable of *real-time reasoning* on first-aid situations — completely offline.
        """
    )

    chatbot = gr.Chatbot(height=500, label="Emergency Chat")
    msg = gr.Textbox(
        label="Type your question about a first-aid situation",
        placeholder="e.g., I burned my hand on hot oil. What should I do?",
    )
    clear = gr.Button("🧹 Clear Chat")

    msg.submit(firstaid_response, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    gr.Markdown(
        """
        **💡 Tip:** Try questions like:
        - *Someone fainted suddenly, what should I do?*
        - *I got bitten by a dog — should I go to the hospital?*
        - *My friend is having a nosebleed that won’t stop.*
        """
    )

# =======================================================
# 4️⃣ Launch
# =======================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
