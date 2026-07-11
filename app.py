import streamlit as st
import pickle
import re
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile
import warnings
from datetime import datetime

# ── API Libraries ──
try:
    import deepl
except:
    deepl = None
try:
    import openai
except:
    openai = None

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="LinguaBridge",
    page_icon="🌍",
    layout="wide"
)

# ── LOAD ML MODELS ──
@st.cache_resource
def load_models():
    lang_model   = pickle.load(open('language_model.pkl', 'rb'))
    tfidf_lang   = pickle.load(open('tfidf_lang.pkl', 'rb'))
    lang_encoder = pickle.load(open('lang_encoder.pkl', 'rb'))
    sent_model   = pickle.load(open('sentiment_model.pkl', 'rb'))
    tfidf_sent   = pickle.load(open('tfidf_sentiment.pkl', 'rb'))
    tone_model   = pickle.load(open('tone_model.pkl', 'rb'))
    tfidf_tone   = pickle.load(open('tfidf_tone.pkl', 'rb'))
    tone_encoder = pickle.load(open('tone_encoder.pkl', 'rb'))
    return (lang_model, tfidf_lang, lang_encoder,
            sent_model, tfidf_sent,
            tone_model, tfidf_tone, tone_encoder)

(lang_model, tfidf_lang, lang_encoder,
 sent_model, tfidf_sent,
 tone_model, tfidf_tone, tone_encoder) = load_models()

# ── API KEYS from Streamlit Secrets ──
DEEPL_API_KEY = st.secrets.get("DEEPL_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# Initialize DeepL
deepl_translator = deepl.Translator(DEEPL_API_KEY) if DEEPL_API_KEY and deepl else None

# Initialize OpenAI
openai_client = None
openai_legacy = False

if OPENAI_API_KEY and openai:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except AttributeError:
        openai.api_key = OPENAI_API_KEY
        openai_client = True
        openai_legacy = True
    except Exception as e:
        st.warning(f"OpenAI init error: {str(e)}")

# ── UI LABELS ──
UI_LABELS = {
    "English": {
        "your_lang": "Your Language:",
        "write_here": "Type your message here...",
        "send_to_b": "🔄 Send to Person B",
        "send_to_a": "🔄 Send to Person A",
        "received": "📨 Received:",
        "analysis": "📊 Analysis:",
        "language": "🌍 Language",
        "sentiment": "💭 Sentiment",
        "tone": "🎭 Tone",
        "warning": "⚠️ Please write something first!",
        "kb_guide": "💡 Tip: Press Windows+Space to change keyboard",
        "spelling": "✏️ Spelling Check",
        "correct": "✅ Suggestion:",
    },
    "Urdu": {
        "your_lang": "آپ کی زبان:",
        "write_here": "اپنا پیغام یہاں ٹائپ کریں...",
        "send_to_b": "🔄 دوسرے کو بھیجیں",
        "send_to_a": "🔄 پہلے کو بھیجیں",
        "received": "📨 موصول ہوا:",
        "analysis": "📊 تجزیہ:",
        "language": "🌍 زبان",
        "sentiment": "💭 جذبات",
        "tone": "🎭 لہجہ",
        "warning": "⚠️ پہلے کچھ لکھیں!",
        "kb_guide": "💡 مشورہ: کی بورڈ تبدیل کرنے کے لیے Windows+Space دبائیں",
        "spelling": "✏️ ہجے کی جانچ",
        "correct": "✅ تجویز:",
    },
    # Add more languages as needed...
}

def get_label(lang, key):
    labels = UI_LABELS.get(lang, UI_LABELS["English"])
    return labels.get(key, UI_LABELS["English"][key])

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\'\"]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text):
    X = tfidf_lang.transform([preprocess(text)])
    return lang_encoder.inverse_transform(lang_model.predict(X))[0]

def detect_sentiment(text):
    X = tfidf_sent.transform([preprocess(text)])
    pred = sent_model.predict(X)[0]
    return "Positive 😊" if pred == 1 else "Negative 😔"

def detect_tone(text):
    X = tfidf_tone.transform([preprocess(text)])
    return tone_encoder.inverse_transform(tone_model.predict(X))[0].upper()

def text_to_speech(text, lang):
    try:
        code = GTTS_CODES.get(lang, "en")
        tts = gTTS(text=text, lang=code, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(tmp.name)
        return tmp.name
    except:
        return None

# ── LANGUAGES ──
LANG_CODES = {
    "English": "en", "Urdu": "ur", "French": "fr", "Spanish": "es",
    "Arabic": "ar", "German": "de", "Hindi": "hi", "Italian": "it",
    "Russian": "ru", "Turkish": "tr", "Dutch": "nl", "Greek": "el",
    "Swedish": "sv", "Danish": "da", "Tamil": "ta", "Malayalam": "ml",
    "Kannada": "kn", "Portuguese": "pt", "Chinese": "zh-cn",
    "Japanese": "ja", "Korean": "ko", "Persian": "fa", "Pashto": "ps",
    "Kurdish": "ku", "Bengali": "bn", "Punjabi": "pa", "Marathi": "mr",
    "Gujarati": "gu", "Telugu": "te", "Sindhi": "sd", "Nepali": "ne",
    "Sinhala": "si", "Indonesian": "id", "Malay": "ms", "Tagalog": "tl",
    "Swahili": "sw", "Yoruba": "yo", "Zulu": "zu", "Xhosa": "xh",
    "Hausa": "ha", "Amharic": "am", "Hebrew": "he", "Maltese": "mt",
    "Mongolian": "mn", "Georgian": "ka", "Armenian": "hy", "Albanian": "sq",
    "Irish": "ga", "Welsh": "cy", "Lithuanian": "lt", "Latvian": "lv",
    "Estonian": "et", "Finnish": "fi", "Hungarian": "hu", "Norwegian": "no",
    "Icelandic": "is", "Afrikaans": "af", "Belarusian": "be", "Ukrainian": "uk",
    "Czech": "cs", "Slovak": "sk", "Bulgarian": "bg", "Serbian": "sr",
    "Croatian": "hr", "Slovenian": "sl", "Bosnian": "bs", "Macedonian": "mk",
    "Catalan": "ca", "Galician": "gl", "Romanian": "ro", "Latin": "la",
    "Vietnamese": "vi", "Thai": "th", "Burmese": "my", "Khmer": "km",
    "Lao": "lo", "Berber": "ber", "Kazakh": "kk", "Uzbek": "uz",
    "Azerbaijani": "az", "Turkmen": "tk", "Kyrgyz": "ky", "Tatar": "tt",
    "Javanese": "jv", "Sundanese": "su",
}

GTTS_CODES = {
    "English": "en", "Urdu": "ur", "French": "fr", "Spanish": "es",
    "Arabic": "ar", "German": "de", "Hindi": "hi", "Italian": "it",
    "Russian": "ru", "Turkish": "tr", "Dutch": "nl", "Greek": "el",
    "Swedish": "sv", "Danish": "da", "Tamil": "ta", "Malayalam": "ml",
    "Kannada": "kn", "Portuguese": "pt", "Chinese": "zh",
    "Japanese": "ja", "Korean": "ko", "Persian": "fa", "Pashto": "ps",
    "Bengali": "bn", "Punjabi": "pa", "Indonesian": "id", "Swahili": "sw",
    "Hebrew": "he", "Norwegian": "no", "Icelandic": "is", "Afrikaans": "af",
    "Ukrainian": "uk", "Czech": "cs", "Slovak": "sk", "Bulgarian": "bg",
    "Serbian": "sr", "Croatian": "hr", "Slovenian": "sl", "Catalan": "ca",
    "Galician": "gl", "Romanian": "ro", "Latin": "la", "Vietnamese": "vi",
    "Thai": "th", "Burmese": "my", "Khmer": "km", "Lao": "lo",
}

# ── SPELL CHECK ──
def check_spelling(text, lang_code="en"):
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language=lang_code)
        words = text.split()
        misspelled = {}
        common_words = {
            "don't": "don't", "don’t": "don't", "can't": "can't", "can’t": "can't",
            "won't": "won't", "won’t": "won't", "shouldn't": "shouldn't",
            "wouldn't": "wouldn't", "couldn't": "couldn't", "isn't": "isn't",
            "aren't": "aren't", "wasn't": "wasn't", "weren't": "weren't",
            "haven't": "haven't", "hasn't": "hasn't", "hadn't": "hadn't",
            "doesn't": "doesn't", "didn't": "didn't", "I'm": "I'm", "I’m": "I'm",
            "you're": "you're", "you’re": "you're", "we're": "we're",
            "they're": "they're", "he's": "he's", "she's": "she's",
            "it's": "it's", "it’s": "it's", "let's": "let's", "ma'am": "ma'am"
        }
        for word in words:
            if word.lower() in common_words:
                continue
            clean_word = re.sub(r'[^\w\s\']', '', word)
            if clean_word and clean_word.lower() not in spell:
                correction = spell.correction(clean_word)
                if correction and correction != clean_word:
                    misspelled[word] = correction
        return misspelled
    except:
        return {}

# ── TRANSLATION FUNCTIONS ──
def translate_with_deepl(text, target_lang):
    try:
        if not deepl_translator:
            return None
        deepl_lang_map = {
            "English": "EN-US", "French": "FR", "Spanish": "ES",
            "German": "DE", "Italian": "IT", "Russian": "RU",
            "Turkish": "TR", "Dutch": "NL", "Greek": "EL",
            "Swedish": "SV", "Danish": "DA", "Portuguese": "PT",
            "Polish": "PL", "Romanian": "RO", "Czech": "CS",
            "Ukrainian": "UK", "Hungarian": "HU", "Finnish": "FI",
            "Norwegian": "NB", "Arabic": "AR", "Hindi": "HI",
            "Urdu": "UR", "Chinese": "ZH", "Japanese": "JA",
            "Korean": "KO"
        }
        lang = deepl_lang_map.get(target_lang, target_lang[:2].upper())
        result = deepl_translator.translate_text(text, target_lang=lang)
        return result.text
    except:
        return None

def translate_with_chatgpt(text, target_lang, source_lang="auto"):
    try:
        if not openai_client:
            return None
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Keep it natural and conversational. Only return the translated text, nothing else."},
            {"role": "user", "content": text}
        ]
        if not openai_legacy:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
    except:
        return None

def translate_with_google(text, target_lang, source_lang=None):
    """Translate using deep_translator (Google Translate) - Always works on Cloud"""
    try:
        if source_lang is None:
            if any('\u0600' <= c <= '\u06FF' for c in text):
                source_lang = "ur"
            else:
                source_lang = "en"
        code = LANG_CODES.get(target_lang, "en")
        return GoogleTranslator(
            source=source_lang,
            target=code
        ).translate(text)
    except Exception as e:
        return None

def post_process_translation(text, source_lang, target_lang):
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])([a-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def smart_translate(text, target_lang, source_lang=None):
    if source_lang is None:
        if any('\u0600' <= c <= '\u06FF' for c in text):
            source_lang = "ur"
        else:
            source_lang = "en"
    
    results = []
    
    with st.spinner("🔄 Translating..."):
        # 1. Google Translate (Always works on Cloud)
        google_result = translate_with_google(text, target_lang, source_lang)
        if google_result:
            results.append(("Google Translate", google_result))
        
        # 2. ChatGPT (if available)
        chatgpt_result = translate_with_chatgpt(text, target_lang, source_lang)
        if chatgpt_result:
            results.append(("ChatGPT", chatgpt_result))
        
        # 3. DeepL (if available)
        deepl_result = translate_with_deepl(text, target_lang)
        if deepl_result:
            results.append(("DeepL", deepl_result))
    
    if not results:
        return "Translation failed. Please check internet connection.", "Error"
    
    # Select best translation (first available)
    selected_api, selected_text = results[0]
    selected_text = post_process_translation(selected_text, source_lang, target_lang)
    st.caption(f"✅ Translated using: **{selected_api}**")
    return selected_text, selected_api

# ── Session State ──────────────────────
if 'translation_a' not in st.session_state:
    st.session_state.translation_a = ""
if 'translation_b' not in st.session_state:
    st.session_state.translation_b = ""
if 'analysis_a' not in st.session_state:
    st.session_state.analysis_a = None
if 'analysis_b' not in st.session_state:
    st.session_state.analysis_b = None

# ── UI ─────────────────────────────────
st.title("🌍 LinguaBridge")
st.subheader("Real-Time Multilingual Interpreter")
st.divider()

st.info("🔄 **Translation APIs:** Google Translate (Stable) + DeepL + ChatGPT")

with st.expander("⌨️ Keyboard Language Change Guide", expanded=False):
    st.info("""
    **To change keyboard language in Windows:**
    1. Press **Windows Key + Space Bar** simultaneously
    2. Keep pressing Space to cycle through installed languages
    3. Select your preferred language
    """)

st.divider()

lang_options = sorted(list(LANG_CODES.keys()))
col1, col2 = st.columns(2)

# ── PERSON A ───────────────────────────
with col1:
    st.markdown("### 👤 Person A")
    lang_a = st.selectbox(get_label("English", "your_lang"), lang_options, index=lang_options.index("English") if "English" in lang_options else 0, key="lang_a")
    st.caption(get_label(lang_a, "kb_guide"))
    text_a = st.text_area("", height=120, key="input_a", placeholder=get_label(lang_a, "write_here"), label_visibility="collapsed")
    
    if text_a.strip():
        spell_errors = check_spelling(text_a, "en")
        if spell_errors:
            st.warning("✏️ " + get_label(lang_a, "spelling") + ":")
            for wrong, correct in spell_errors.items():
                st.info(f"❌ '{wrong}' → ✅ '{correct}'")
    
    btn_a = st.button(get_label(lang_a, "send_to_b"), use_container_width=True, key="btn_a")
    
    if st.session_state.translation_b:
        st.markdown("---")
        st.markdown(f"**{get_label(lang_a, 'received')}**")
        st.info(st.session_state.translation_b)
        audio_file = text_to_speech(st.session_state.translation_b, lang_a)
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
            try:
                os.unlink(audio_file)
            except:
                pass
    
    if st.session_state.analysis_b:
        a = st.session_state.analysis_b
        st.markdown(f"**{get_label(lang_a, 'analysis')}**")
        c1, c2, c3 = st.columns(3)
        c1.metric(get_label(lang_a, "language"), a['lang'])
        c2.metric(get_label(lang_a, "sentiment"), a['sent'])
        c3.metric(get_label(lang_a, "tone"), a['tone'])

# ── PERSON B ───────────────────────────
with col2:
    st.markdown("### 👤 Person B")
    lang_b = st.selectbox(get_label("English", "your_lang"), lang_options, index=lang_options.index("Urdu") if "Urdu" in lang_options else 1, key="lang_b")
    st.caption(get_label(lang_b, "kb_guide"))
    text_b = st.text_area("", height=120, key="input_b", placeholder=get_label(lang_b, "write_here"), label_visibility="collapsed")
    
    if text_b.strip():
        spell_errors = check_spelling(text_b, "en")
        if spell_errors:
            st.warning("✏️ " + get_label(lang_b, "spelling") + ":")
            for wrong, correct in spell_errors.items():
                st.info(f"❌ '{wrong}' → ✅ '{correct}'")
    
    btn_b = st.button(get_label(lang_b, "send_to_a"), use_container_width=True, key="btn_b")
    
    if st.session_state.translation_a:
        st.markdown("---")
        st.markdown(f"**{get_label(lang_b, 'received')}**")
        st.info(st.session_state.translation_a)
        audio_file = text_to_speech(st.session_state.translation_a, lang_b)
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
            try:
                os.unlink(audio_file)
            except:
                pass
    
    if st.session_state.analysis_a:
        a = st.session_state.analysis_a
        st.markdown(f"**{get_label(lang_b, 'analysis')}**")
        c1, c2, c3 = st.columns(3)
        c1.metric(get_label(lang_b, "language"), a['lang'])
        c2.metric(get_label(lang_b, "sentiment"), a['sent'])
        c3.metric(get_label(lang_b, "tone"), a['tone'])

# ── TRANSLATION LOGIC ──────────────────
if btn_a:
    final_text_a = text_a.strip()
    if final_text_a:
        source_lang = "ur" if any('\u0600' <= c <= '\u06FF' for c in final_text_a) else "en"
        translation_a, used_api = smart_translate(final_text_a, lang_b, source_lang)
        analysis_a = {
            'lang': detect_language(final_text_a),
            'sent': detect_sentiment(final_text_a),
            'tone': detect_tone(final_text_a)
        }
        st.session_state.translation_a = translation_a
        st.session_state.analysis_a = analysis_a
        st.rerun()
    else:
        st.warning(get_label(lang_a, "warning"))

if btn_b:
    final_text_b = text_b.strip()
    if final_text_b:
        source_lang = "ur" if any('\u0600' <= c <= '\u06FF' for c in final_text_b) else "en"
        translation_b, used_api = smart_translate(final_text_b, lang_a, source_lang)
        analysis_b = {
            'lang': detect_language(final_text_b),
            'sent': detect_sentiment(final_text_b),
            'tone': detect_tone(final_text_b)
        }
        st.session_state.translation_b = translation_b
        st.session_state.analysis_b = analysis_b
        st.rerun()
    else:
        st.warning(get_label(lang_b, "warning"))

st.divider()
st.caption("🌍 LinguaBridge — Breaking Language Barriers! | 💬 Text | 🗣️ 100+ Languages | ✏️ Spell Check | 🔊 Audio")
