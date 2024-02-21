from transformers import pipeline
classifier = pipeline(task="sentiment-analysis")
result = classifier("I am so happy that I am learning Hugging Face!")
print(result)
# Output: {'label': 'POSITIVE', 'score': 0.9998691082000732}]