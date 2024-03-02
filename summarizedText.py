from flask import Flask, request, jsonify
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
model.to('cuda')

@app.route('/api/greeting', methods=['GET'])
def get_greeting():
    return 'Hi from Python'
    
# Endpoint for text summarization
@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        # Get the input text from the request body
        input_text = request.json['text']

        # Tokenize and generate summary
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the summary and return it
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({'summarized_text': summarized_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
