from transformers import AutoTokenizer
import os 
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokens = []

# this helped me convert all of the files in that directory to .txt files
# for root, dirs, files in os.walk("2of2"):
#     for f in files:
#         file_path = os.path.join(root, f)
#         if "wiki_" in file_path:
#            new_path = file_path + ".txt"
#            os.rename(file_path, new_path)

for root, dirs, files in os.walk("1of2"):
    for f in files:
        file_path = os.path.join(root, f)
        if "wiki_" in file_path:
            with open(file_path, "r", encoding="utf8") as f:
                for line in f:
                    tokens.append(tokenizer(line))


print(tokens)

