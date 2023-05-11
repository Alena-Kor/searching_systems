from transformers.utils import logging
logging.set_verbosity(40)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
#DEVICE = torch.device("cuda:0")

# Загружаем модель ruGPT от сбера
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# возьмем какой-нибудь текст
text = 'Он на первый взгляд, самый обычный среднестатистический американец'
input_ids = tokenizer.encode(text, return_tensors="pt")

for temp in [0.1, 0.2, 0.3, 0.5, 0.7, 1., 2., 5.]:
    for top_k in [3, 10, 30, 50]:
        out = model.generate(input_ids,
                     do_sample=True,
                     temperature=temp,
                     top_k=top_k,
                     max_length=100,
                     )

        generated_text = list(map(tokenizer.decode, out))[0]
        print("### text with temp, top_k - ", temp, top_k)
        print(generated_text)
        print()

# beam search уже реализован в hg поэтому нужно только задать параметр num_beams
# out = model.generate(input_ids, do_sample=True, num_beams=5, top_k=0, max_length=60)
#
# generated_text = list(map(tokenizer.decode, out))[0]
# print()
# print(generated_text.replace('<s>', ' '))