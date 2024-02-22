from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

# Token classification or Named Entity Recognition (NER)
diagnostic_description = """
A 48 year-old female presented with vaginal bleeding and abnormal Pap smears.
She was diagnosed with stage 1B1 cervical cancer and underwent a radical hysterectomy and lymph node dissection.
She was treated with adjuvant radiation therapy and chemotherapy.
She is currently without evidence of disease."""

ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ner_diagn_result = ner_pipe(diagnostic_description)
# print(ner_diagn_result)
for entity in ner_diagn_result:
    word = entity['word']
    category = entity['entity_group'] if entity.get('entity_group') else ''
    score = entity['score']
    if not '##' in word:
        print(f"{word} [{category}]")