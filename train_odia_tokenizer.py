from odiatokenizer_util import OdiaBPETokenizer
import time

def load_telugu_texts(file_paths):
    texts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

# Training
data_files = [
    "Odia/odia_tokenizers_test.vocab",
    # "../A11_Odia_Tokenizer/odia_sample.txt",
]

# Create and train tokenizer
tokenizer = OdiaBPETokenizer(vocab_size=5000)
texts = load_telugu_texts(data_files)

# 3,000,000 tested already, 48 mins run time
texts[0] = texts[0][:5000000] 


# print(f"Number of base_vocab: {len(tokenizer.base_vocab)}")
# print(f"base_vocab: {tokenizer.base_vocab}")



start_time = time.time()
tokenizer.train(texts, min_freq=2)
end_time = time.time()
print(f"Time taken to train: {end_time - start_time} seconds")

# Test the tokenizer
test_text = "ଦରକାର ସମୟରେ ସାହାଯ୍ୟ ପ୍ରକୃତରେ ସାହାଯ୍ୟ ।"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

# Calculate compression ratio
compression_ratio = tokenizer.calculate_compression_ratio(test_text)

print(f"Original text: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Compression ratio: {compression_ratio:.2f}")

# Print some statistics
print(f"\nVocabulary size: {len(tokenizer.vocab)}")
print(f"Number of merges: {len(tokenizer.merges)}")

# print("\nSample merges:")
# for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
#     print(f"{pair} -> {merged}")

# Save the tokenizer
tokenizer.save("telugu_bpe_tokenizer.json") 