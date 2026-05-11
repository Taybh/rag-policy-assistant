from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2AnswerGenerator:
    def __init__(self, model_name, max_input_length=1024, max_new_tokens=80):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

    def generate_answer(self, question, contexts):
        context_text = "\n".join(contexts)

        prompt = f"""
Use the following company policy context to answer the employee question.

Context:
{context_text}

Question:
{question}

Answer:
""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )

        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return answer

