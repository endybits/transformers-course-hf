from transformers import pipeline

def classify_sentiment(text):
    classifier = pipeline(task="sentiment-analysis")
    result = classifier(text)
    return result

text = [
    "I am so happy that I am learning Hugging Face!",
    "I hate the new transformers movie"

]
result = classify_sentiment(text)
print(result)

# Output: [{'label': 'POSITIVE', 'score': 0.9998691082000732}, {'label': 'NEGATIVE', 'score': 0.9993649125099182}]