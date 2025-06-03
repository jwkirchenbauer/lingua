# here we load a hf model and test it on a simple prompt

from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompt: The capital of France is

MODEL_PATH = "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lin_hq_cd_128N/avg_checkpoint"
# Generated text: The capital of France is Paris, and its largest city is Lyon. It is the world's third most populous country, with a population of around 66 million people. France is a developed country with a high standard of living. It is a
# Generated text: The capital of France is Paris, which is located in the north of the country. Paris is the largest city in France and one of the largest cities in Europe. It is known for its beautiful architecture and cultural heritage. The other

# MODEL_PATH = "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lin_hq_cd_128N/checkpoints_hf/0000239000"
# Generated text: The capital of France is Paris. It is also the capital of the Île-de-France, which is an area around Paris, including other smaller cities, which comprise the conurbation of Greater Paris. The largest city in

# MODEL_PATH = "/p/lustre5/kirchenb/common-pile-root/lingua/output/prod_lingua_7B_2T_lin_hq_cd_128N/checkpoints_hf/0000235000"
# Generated text: The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris.

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print(f"Model loaded from {MODEL_PATH}")
print(model)


def generate_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]
    # this gives an error
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "The capital of France is"
print(f"Prompt: {prompt}")
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")

breakpoint()
