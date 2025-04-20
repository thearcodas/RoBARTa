from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os, json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


import torch
import os
import io
import base64
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document

# Initialize models
tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_roberta = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=model_roberta, tokenizer=tokenizer_roberta)

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

def summarizer(text):
# Summarize text using BART and t5 model
    input_ids = bart_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    bart_summary_ids = bart_model.generate(input_ids, max_length=200, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    bart_summary = bart_tokenizer.decode(bart_summary_ids[0], skip_special_tokens=True)

    input_text = "summarize: " + bart_summary
    t5_input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    t5_summary_ids = t5_model.generate(t5_input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    t5_summary = t5_tokenizer.decode(t5_summary_ids[0], skip_special_tokens=True)
    return t5_summary

def sentiment(text):
    return sentiment_pipeline(text)
@csrf_exempt
def summarize_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text')
        
        summary= summarizer(text)
        return JsonResponse({"summary": summary})

    return JsonResponse({"error": "Invalid request"}, status=400)

@csrf_exempt
def sentiment_view(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            text = body.get("text", "")
            if not text:
                return JsonResponse({"error": "Text is required."}, status=400)

            # Get prediction
            results = sentiment(text)
            label = results[0]['label']
            score = results[0]['score']

            explanation = {
                "LABEL_0": "The sentiment is negative. This may indicate dissatisfaction, criticism, or disappointment in the text.",
                "LABEL_1": "The sentiment is neutral. The text appears to be objective or balanced without strong emotional expression.",
                "LABEL_2": "The sentiment is positive. It conveys approval, satisfaction, or optimism."
            }

            return JsonResponse({
                "label": label,
                "score": score,
                "explanation": explanation.get(label, "No explanation available for this label.")
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)


@csrf_exempt
def upload_file_view(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            uploaded_file = request.FILES['file']
            filename = uploaded_file.name.lower()

            if filename.endswith('.pdf'):
                reader = PdfReader(uploaded_file)
                extracted_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif filename.endswith('.docx'):
                doc = Document(uploaded_file)
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
            elif filename.endswith('.txt'):
                extracted_text = uploaded_file.read().decode('utf-8')
            else:
                return JsonResponse({"error": "Unsupported file format. Please upload a PDF, DOCX, or TXT file."}, status=400)

            return JsonResponse({"extracted_text": extracted_text})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "No file uploaded or invalid request method."}, status=400)

@csrf_exempt
def export_pdf_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        content = data.get("content")
        chart_data = data.get("chart")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)

        if chart_data:
            image_data = base64.b64decode(chart_data)
            with open("chart.png", "wb") as f:
                f.write(image_data)
            pdf.image("chart.png", x=10, y=pdf.get_y(), w=100)

        output_path = os.path.join(default_storage.location, "output.pdf")
        pdf.output(output_path)

        with open(output_path, 'rb') as f:
            pdf_bytes = f.read()

        return JsonResponse({"pdf": base64.b64encode(pdf_bytes).decode('utf-8')})

    return JsonResponse({"error": "Invalid request"}, status=400)




def index_view(request):
    return render(request, 'index.html')
