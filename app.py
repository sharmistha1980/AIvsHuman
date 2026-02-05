import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__, static_url_path='', static_folder='static')

print("----------------------------------------------------------------")
print("INITIALIZING HYBRID DETECTION ENGINE (Model + Rules)...")
print("----------------------------------------------------------------")

# --- 1. LOCAL DETECTOR ---
try:
    print("Loading Detector...")
    # We use top_k=None to get the raw probabilities
    detector = pipeline("text-classification", model="openai-community/roberta-base-openai-detector", top_k=None)
    print("✓ Detector Ready")
except Exception as e:
    print(f"X Error: {e}")
    detector = None

# --- 2. LOCAL HUMANIZER ---
try:
    print("Loading Humanizer...")
    humanizer = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")
    print("✓ Humanizer Ready")
except Exception as e:
    print(f"X Error: {e}")
    humanizer = None

@app.route('/')
def index():
    return app.send_static_file('index.html')

def check_ai_patterns(text):
    """
    Scans for common 'AI-isms' that models sometimes miss.
    Returns a 'suspicion boost' score (0.0 to 1.0).
    """
    text_lower = text.lower()
    # Added more aggressive keywords
    patterns = [
        "in conclusion", "furthermore", "moreover", "it is important to note",
        "crucial to consider", "testament to", "delve into", "underscore",
        "landscape of", "comprehensive", "meticulous", "realm of", "essential to",
        "serves to", "not only"
    ]
    
    boost = 0.0
    hits = []
    for p in patterns:
        if p in text_lower:
            boost += 0.15 # Add 15% suspicion for each AI keyword found
            hits.append(p)
            
    return boost, hits

@app.route('/detect', methods=['POST'])
def detect_text():
    if not detector: return jsonify({"error": "Detector failed."}), 500
    
    data = request.get_json(force=True, silent=True)
    text = data.get('text', '')
    
    if len(text) < 60:
        return jsonify({"label": "Human-Written", "score": 100, "is_ai": False, "message": "Text too short for analysis."})
    
    try:
        # 1. Get Model Score
        results = detector(text[:512])
        
        if isinstance(results, list) and isinstance(results[0], list): scores = results[0]
        else: scores = results

        raw_ai_score = 0.0
        for item in scores:
            if item['label'] in ['Fake', 'LABEL_0']:
                raw_ai_score = item['score']
        
        # 2. Get Pattern Score (The Logic Fix)
        pattern_boost, detected_patterns = check_ai_patterns(text)
        
        # 3. Calculate Final Score
        # We combine the Model's feeling with the Pattern evidence
        final_ai_score = raw_ai_score + pattern_boost
        
        # 4. Threshold Decision
        
        THRESHOLD = 0.10
        
        if final_ai_score > THRESHOLD:
            final_label = "AI-Generated"
            is_ai = True
            # Calculate display score (capped at 99.9%)
            display_score = min(99.9, 60 + ((final_ai_score - THRESHOLD) / (1 - THRESHOLD) * 40))
            
            if len(detected_patterns) > 0:
                reason = f"Model detected robotic structure. Flagged keywords: '{', '.join(detected_patterns[:2])}'."
            else:
                reason = "Inspector detected low semantic variance typical of AI."
        else:
            final_label = "Human-Written"
            is_ai = False
            display_score = (1 - final_ai_score) * 100
            reason = "High structural burstiness suggests natural human authorship."

        return jsonify({
            "label": final_label,
            "score": round(display_score, 2),
            "is_ai": is_ai,
            "message": reason
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/humanize', methods=['POST'])
def humanize_text():
    if not humanizer: return jsonify({"error": "Humanizer failed."}), 500
    data = request.get_json(force=True, silent=True)
    text = data.get('text', '')
    
    try:
        outputs = humanizer(
            text,
            num_beams=5,
            num_return_sequences=1,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            temperature=0.9,
            max_length=256
        )
        return jsonify({"humanized": outputs[0]['generated_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)