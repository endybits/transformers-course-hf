from transformers import pipeline

unmasker = pipeline("fill-mask")
mask_filled_list = unmasker("Sydney is an <mask> city.")
mask_filled_list.sort(key=lambda x: x['score'], reverse=True)
for item in mask_filled_list:
    print(item['sequence'], item['score'])