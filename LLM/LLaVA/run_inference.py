import threading
import argparse
from PIL import Image
from io import BytesIO
import re
from flask import Flask, request
from local_llm import LocalLM, ChatHistory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='liuhaotian/llava-v1.5-7b', type=str, help="either the path to the model, or HuggingFace model repo/name")
    parser.add_argument('--quant', default=None, type=str, help="for AWQ or MLC, either specify the quantization method, or the path to the quantized model AWQ and MLC API only")
    parser.add_argument('--api', default='mlc', type=str, help="the model backend API to use:  'auto_gptq', 'awq', 'mlc', or 'hf'if left as None, it will attempt to be automatically determined.")
    parser.add_argument('--vision_model', default=None, type=str, help="for VLMs, override the vision embedding model (CLIP) otherwise, it will use the CLIP variant from the config.")
    parser.add_argument('--port', default=5001, type=int, help="The port number of the service program")
    _args = parser.parse_args()
    return _args


class Inference():
    def __init__(self, args) -> None:
        self.model = LocalLM.from_pretrained(
            args.model,
            quant=args.quant,
            api=args.api,
            vision_model=args.vision_model,
        )

        self.app = Flask(__name__)
        self.setup_routes()
    
        self.lock = threading.Lock()

    def predict(self,
        user_text,
        image_path,
        system_prompt,
        max_new_tokens=1024,
        min_new_tokens=-1,
        do_sample=False,
        repetition_penalty=1.0,
        temperature=0.7,
        top_p=0.95,
    ):
        print(image_path, flush=True)
        chat_history = ChatHistory(self.model, chat_template=None, system_prompt=system_prompt)
        entry = chat_history.append(role="user", msg=image_path)
        entry = chat_history.append(role="user", msg=user_text)
        if "image" in entry and "text" not in entry:
            return "only image message, waiting for user prompt"
        embedding, position = chat_history.embed_chat()
        reply = self.model.generate(
            embedding,
            streaming=False,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            do_sample=bool(do_sample),
            repetition_penalty=float(repetition_penalty),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        return reply
    
    def is_chinese(self, text):
        pattern = re.compile(r"[\u4e00-\u9fff]")
        return bool(pattern.search(text))

    def setup_routes(self):

        @self.app.route("/v2/generate", methods=["POST"])
        def generate2():
            user_prompt = request.form.get("user_prompt") or "describe the image."
            if self.is_chinese(user_prompt):
                return "only english please."
            sys_prompt = (
                request.form.get("sys_prompt") or "you are combo AI robot, will think how to finish human question"
            )
            max_new_tokens = request.form.get("max_new_tokens") or 1024
            min_new_tokens = request.form.get("min_new_tokens") or -1
            do_sample = request.form.get("do_sample") or False
            repetition_penalty = request.form.get("repetition_penalty") or 1.0
            temperature = request.form.get("temperature") or 0.7
            top_p = request.form.get("top_p") or 0.95
            with self.lock:
                file = request.files["image"]
                file_data = file.read()
                image = Image.open(BytesIO(file_data))
                output = self.predict(
                    user_prompt,
                    image,
                    sys_prompt,
                    max_new_tokens,
                    min_new_tokens,
                    do_sample,
                    repetition_penalty,
                    temperature,
                    top_p,
                )
                print(output)
                return output

if __name__ == "__main__":
    args = parse_args()
    runner = Inference(args)
    runner.app.run(host="0.0.0.0", port=args.port)


