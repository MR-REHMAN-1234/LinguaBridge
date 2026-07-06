import streamlit as st
import pickle
import re
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
    lang_model = pickle.load(open('language_model.pkl', 'rb'))
    tfidf_lang = pickle.load(open('tfidf_lang.pkl', 'rb'))
    lang_encoder = pickle.load(open('lang_encoder.pkl', 'rb'))
    sent_model = pickle.load(open('sentiment_model.pkl', 'rb'))
    tfidf_sent = pickle.load(open('tfidf_sentiment.pkl', 'rb'))
    tone_model = pickle.load(open('tone_model.pkl', 'rb'))
    tfidf_tone = pickle.load(open('tfidf_tone.pkl', 'rb'))
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

# ── UI LABELS ── (30+ Languages)
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
    },
    "French": {
        "your_lang": "Votre langue:",
        "write_here": "Tapez votre message ici...",
        "send_to_b": "🔄 Envoyer à B",
        "send_to_a": "🔄 Envoyer à A",
        "received": "📨 Reçu:",
        "analysis": "📊 Analyse:",
        "language": "🌍 Langue",
        "sentiment": "💭 Sentiment",
        "tone": "🎭 Ton",
        "warning": "⚠️ Écrivez quelque chose d'abord!",
        "kb_guide": "💡 Astuce: Appuyez sur Windows+Espace pour changer la langue",
    },
    "Spanish": {
        "your_lang": "Tu idioma:",
        "write_here": "Escribe tu mensaje aquí...",
        "send_to_b": "🔄 Enviar a B",
        "send_to_a": "🔄 Enviar a A",
        "received": "📨 Recibido:",
        "analysis": "📊 Análisis:",
        "language": "🌍 Idioma",
        "sentiment": "💭 Sentimiento",
        "tone": "🎭 Tono",
        "warning": "⚠️ Escribe algo primero!",
        "kb_guide": "💡 Consejo: Presiona Windows+Espacio para cambiar teclado",
    },
    "Arabic": {
        "your_lang": "لغتك:",
        "write_here": "اكتب رسالتك هنا...",
        "send_to_b": "🔄 إرسال إلى B",
        "send_to_a": "🔄 إرسال إلى A",
        "received": "📨 تم الاستلام:",
        "analysis": "📊 التحليل:",
        "language": "🌍 اللغة",
        "sentiment": "💭 المشاعر",
        "tone": "🎭 النبرة",
        "warning": "⚠️ اكتب شيئاً أولاً!",
        "kb_guide": "💡 تلميح: اضغط على Windows+Space لتغيير لوحة المفاتيح",
    },
    "German": {
        "your_lang": "Ihre Sprache:",
        "write_here": "Geben Sie hier Ihre Nachricht ein...",
        "send_to_b": "🔄 An Person B senden",
        "send_to_a": "🔄 An Person A senden",
        "received": "📨 Erhalten:",
        "analysis": "📊 Analyse:",
        "language": "🌍 Sprache",
        "sentiment": "💭 Gefühl",
        "tone": "🎭 Ton",
        "warning": "⚠️ Bitte schreiben Sie zuerst etwas!",
        "kb_guide": "💡 Tipp: Drücken Sie Windows+Leertaste zum Wechseln",
    },
    "Hindi": {
        "your_lang": "आपकी भाषा:",
        "write_here": "अपना संदेश यहाँ टाइप करें...",
        "send_to_b": "🔄 B को भेजें",
        "send_to_a": "🔄 A को भेजें",
        "received": "📨 प्राप्त हुआ:",
        "analysis": "📊 विश्लेषण:",
        "language": "🌍 भाषा",
        "sentiment": "💭 भावना",
        "tone": "🎭 स्वर",
        "warning": "⚠️ पहले कुछ लिखें!",
        "kb_guide": "💡 सुझाव: कीबोर्ड बदलने के लिए Windows+Space दबाएँ",
    },
    "Italian": {
        "your_lang": "La tua lingua:",
        "write_here": "Scrivi il tuo messaggio qui...",
        "send_to_b": "🔄 Invia a B",
        "send_to_a": "🔄 Invia a A",
        "received": "📨 Ricevuto:",
        "analysis": "📊 Analisi:",
        "language": "🌍 Lingua",
        "sentiment": "💭 Sentimento",
        "tone": "🎭 Tono",
        "warning": "⚠️ Scrivi qualcosa prima!",
        "kb_guide": "💡 Suggerimento: Premi Windows+Spazio per cambiare tastiera",
    },
    "Russian": {
        "your_lang": "Ваш язык:",
        "write_here": "Введите ваше сообщение здесь...",
        "send_to_b": "🔄 Отправить B",
        "send_to_a": "🔄 Отправить A",
        "received": "📨 Получено:",
        "analysis": "📊 Анализ:",
        "language": "🌍 Язык",
        "sentiment": "💭 Настроение",
        "tone": "🎭 Тон",
        "warning": "⚠️ Сначала напишите что-нибудь!",
        "kb_guide": "💡 Совет: Нажмите Windows+Пробел для смены раскладки",
    },
    "Turkish": {
        "your_lang": "Diliniz:",
        "write_here": "Mesajınızı buraya yazın...",
        "send_to_b": "🔄 B'ye Gönder",
        "send_to_a": "🔄 A'ya Gönder",
        "received": "📨 Alındı:",
        "analysis": "📊 Analiz:",
        "language": "🌍 Dil",
        "sentiment": "💭 Duygu",
        "tone": "🎭 Ton",
        "warning": "⚠️ Lütfen önce bir şey yazın!",
        "kb_guide": "💡 İpucu: Klavye değiştirmek için Windows+Space basın",
    },
    "Dutch": {
        "your_lang": "Uw taal:",
        "write_here": "Typ uw bericht hier...",
        "send_to_b": "🔄 Stuur naar B",
        "send_to_a": "🔄 Stuur naar A",
        "received": "📨 Ontvangen:",
        "analysis": "📊 Analyse:",
        "language": "🌍 Taal",
        "sentiment": "💭 Gevoel",
        "tone": "🎭 Toon",
        "warning": "⚠️ Schrijf eerst iets!",
        "kb_guide": "💡 Tip: Druk op Windows+Spatie om toetsenbord te wisselen",
    },
    "Greek": {
        "your_lang": "Η γλώσσα σας:",
        "write_here": "Πληκτρολογήστε το μήνυμά σας εδώ...",
        "send_to_b": "🔄 Αποστολή στον B",
        "send_to_a": "🔄 Αποστολή στον A",
        "received": "📨 Λήφθηκε:",
        "analysis": "📊 Ανάλυση:",
        "language": "🌍 Γλώσσα",
        "sentiment": "💭 Συναίσθημα",
        "tone": "🎭 Τόνος",
        "warning": "⚠️ Γράψτε κάτι πρώτα!",
        "kb_guide": "💡 Συμβουλή: Πατήστε Windows+Space για αλλαγή γλώσσας",
    },
    "Swedish": {
        "your_lang": "Ditt språk:",
        "write_here": "Skriv ditt meddelande här...",
        "send_to_b": "🔄 Skicka till B",
        "send_to_a": "🔄 Skicka till A",
        "received": "📨 Mottaget:",
        "analysis": "📊 Analys:",
        "language": "🌍 Språk",
        "sentiment": "💭 Känsla",
        "tone": "🎭 Ton",
        "warning": "⚠️ Skriv något först!",
        "kb_guide": "💡 Tips: Tryck Windows+Space för att byta tangentbord",
    },
    "Danish": {
        "your_lang": "Dit sprog:",
        "write_here": "Skriv din besked her...",
        "send_to_b": "🔄 Send til B",
        "send_to_a": "🔄 Send til A",
        "received": "📨 Modtaget:",
        "analysis": "📊 Analyse:",
        "language": "🌍 Sprog",
        "sentiment": "💭 Følelse",
        "tone": "🎭 Tone",
        "warning": "⚠️ Skriv noget først!",
        "kb_guide": "💡 Tip: Tryk Windows+Mellemrum for at skifte tastatur",
    },
    "Tamil": {
        "your_lang": "உங்கள் மொழி:",
        "write_here": "உங்கள் செய்தியை இங்கே தட்டச்சு செய்யவும்...",
        "send_to_b": "🔄 B க்கு அனுப்பு",
        "send_to_a": "🔄 A க்கு அனுப்பு",
        "received": "📨 பெறப்பட்டது:",
        "analysis": "📊 பகுப்பாய்வு:",
        "language": "🌍 மொழி",
        "sentiment": "💭 உணர்வு",
        "tone": "🎭 தொனி",
        "warning": "⚠️ முதலில் ஏதாவது எழுதுங்கள்!",
        "kb_guide": "💡 குறிப்பு: Windows+Space அழுத்தி விசைப்பலகை மாற்றவும்",
    },
    "Malayalam": {
        "your_lang": "നിങ്ങളുടെ ഭാഷ:",
        "write_here": "നിങ്ങളുടെ സന്ദേശം ഇവിടെ ടൈപ്പ് ചെയ്യുക...",
        "send_to_b": "🔄 B യിലേക്ക് അയയ്ക്കുക",
        "send_to_a": "🔄 A യിലേക്ക് അയയ്ക്കുക",
        "received": "📨 ലഭിച്ചു:",
        "analysis": "📊 വിശകലനം:",
        "language": "🌍 ഭാഷ",
        "sentiment": "💭 വികാരം",
        "tone": "🎭 സ്വരം",
        "warning": "⚠️ ആദ്യം എന്തെങ്കിലും എഴുതുക!",
        "kb_guide": "💡 സഹായം: Windows+Space അമർത്തി കീബോർഡ് മാറ്റുക",
    },
    "Kannada": {
        "your_lang": "ನಿಮ್ಮ ಭಾಷೆ:",
        "write_here": "ನಿಮ್ಮ ಸಂದೇಶವನ್ನು ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ...",
        "send_to_b": "🔄 B ಗೆ ಕಳುಹಿಸು",
        "send_to_a": "🔄 A ಗೆ ಕಳುಹಿಸು",
        "received": "📨 ಸ್ವೀಕರಿಸಲಾಗಿದೆ:",
        "analysis": "📊 ವಿಶ್ಲೇಷಣೆ:",
        "language": "🌍 ಭಾಷೆ",
        "sentiment": "💭 ಭಾವನೆ",
        "tone": "🎭 ಧಾಟಿ",
        "warning": "⚠️ ಮೊದಲು ಏನನ್ನಾದರೂ ಬರೆಯಿರಿ!",
        "kb_guide": "💡 ಸಲಹೆ: Windows+Space ಒತ್ತಿ ಕೀಬೋರ್ಡ್ ಬದಲಾಯಿಸಿ",
    },
    "Portuguese": {
        "your_lang": "Sua língua:",
        "write_here": "Digite sua mensagem aqui...",
        "send_to_b": "🔄 Enviar para B",
        "send_to_a": "🔄 Enviar para A",
        "received": "📨 Recebido:",
        "analysis": "📊 Análise:",
        "language": "🌍 Língua",
        "sentiment": "💭 Sentimento",
        "tone": "🎭 Tom",
        "warning": "⚠️ Escreva algo primeiro!",
        "kb_guide": "💡 Dica: Pressione Windows+Espaço para mudar o teclado",
    },
    "Chinese": {
        "your_lang": "您的语言:",
        "write_here": "在此输入您的消息...",
        "send_to_b": "🔄 发送给B",
        "send_to_a": "🔄 发送给A",
        "received": "📨 已收到:",
        "analysis": "📊 分析:",
        "language": "🌍 语言",
        "sentiment": "💭 情感",
        "tone": "🎭 语气",
        "warning": "⚠️ 请先写些什么!",
        "kb_guide": "💡 提示: 按 Windows+空格 切换键盘",
    },
    "Japanese": {
        "your_lang": "あなたの言語:",
        "write_here": "ここにメッセージを入力...",
        "send_to_b": "🔄 Bさんに送信",
        "send_to_a": "🔄 Aさんに送信",
        "received": "📨 受信しました:",
        "analysis": "📊 分析:",
        "language": "🌍 言語",
        "sentiment": "💭 感情",
        "tone": "🎭 トーン",
        "warning": "⚠️ 最初に何か書いてください!",
        "kb_guide": "💡 ヒント: Windows+Space を押してキーボードを切り替える",
    },
    "Korean": {
        "your_lang": "당신의 언어:",
        "write_here": "여기에 메시지를 입력하세요...",
        "send_to_b": "🔄 B에게 보내기",
        "send_to_a": "🔄 A에게 보내기",
        "received": "📨 받음:",
        "analysis": "📊 분석:",
        "language": "🌍 언어",
        "sentiment": "💭 감정",
        "tone": "🎭 톤",
        "warning": "⚠️ 먼저 뭔가 쓰세요!",
        "kb_guide": "💡 팁: Windows+Space를 눌러 키보드 변경",
    },
    "Persian": {
        "your_lang": "زبان شما:",
        "write_here": "پیام خود را اینجا تایپ کنید...",
        "send_to_b": "🔄 ارسال به B",
        "send_to_a": "🔄 ارسال به A",
        "received": "📨 دریافت شد:",
        "analysis": "📊 تحلیل:",
        "language": "🌍 زبان",
        "sentiment": "💭 احساس",
        "tone": "🎭 لحن",
        "warning": "⚠️ لطفاً ابتدا چیزی بنویسید!",
        "kb_guide": "💡 نکته: برای تغییر صفحه کلید Windows+Space را فشار دهید",
    },
    "Bengali": {
        "your_lang": "আপনার ভাষা:",
        "write_here": "আপনার বার্তা এখানে টাইপ করুন...",
        "send_to_b": "🔄 B-কে পাঠান",
        "send_to_a": "🔄 A-কে পাঠান",
        "received": "📨 প্রাপ্ত:",
        "analysis": "📊 বিশ্লেষণ:",
        "language": "🌍 ভাষা",
        "sentiment": "💭 অনুভূতি",
        "tone": "🎭 সুর",
        "warning": "⚠️ দয়া করে প্রথমে কিছু লিখুন!",
        "kb_guide": "💡 টিপ: কীবোর্ড পরিবর্তন করতে Windows+Space চাপুন",
    },
    "Punjabi": {
        "your_lang": "ਤੁਹਾਡੀ ਭਾਸ਼ਾ:",
        "write_here": "ਆਪਣਾ ਸੁਨੇਹਾ ਇੱਥੇ ਟਾਈਪ ਕਰੋ...",
        "send_to_b": "🔄 B ਨੂੰ ਭੇਜੋ",
        "send_to_a": "🔄 A ਨੂੰ ਭੇਜੋ",
        "received": "📨 ਪ੍ਰਾਪਤ:",
        "analysis": "📊 ਵਿਸ਼ਲੇਸ਼ਣ:",
        "language": "🌍 ਭਾਸ਼ਾ",
        "sentiment": "💭 ਭਾਵਨਾ",
        "tone": "🎭 ਲਹਿਜ਼ਾ",
        "warning": "⚠️ ਕਿਰਪਾ ਕਰਕੇ ਪਹਿਲਾਂ ਕੁਝ ਲਿਖੋ!",
        "kb_guide": "💡 ਟਿਪ: ਕੀਬੋਰਡ ਬਦਲਣ ਲਈ Windows+Space ਦਬਾਓ",
    },
    "Indonesian": {
        "your_lang": "Bahasa Anda:",
        "write_here": "Ketik pesan Anda di sini...",
        "send_to_b": "🔄 Kirim ke B",
        "send_to_a": "🔄 Kirim ke A",
        "received": "📨 Diterima:",
        "analysis": "📊 Analisis:",
        "language": "🌍 Bahasa",
        "sentiment": "💭 Sentimen",
        "tone": "🎭 Nada",
        "warning": "⚠️ Silakan tulis sesuatu terlebih dahulu!",
        "kb_guide": "💡 Tip: Tekan Windows+Space untuk mengganti keyboard",
    },
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
def detect_text_style(text):
    text_lower = text.lower()
    informal_indicators = [
        "dil", "nahi", "kar raha", "wassup", "lol", "bro", "bestie",
        "ngl", "tbh", "omg", "fr", "slay", "vibes", "lowkey", "highkey",
        "mera", "tera", "tum", "apna", "hai na", "na", "yaar",
        "gonna", "wanna", "gotta", "kinda", "sorta", "ain't", "y'all"
    ]
    idioms = [
        "dil nahi", "dil karna", "khana khane ka dil",
        "maza aa gaya", "kya baat hai", "kya scene hai"
    ]
    for idiom in idioms:
        if idiom in text_lower:
            return "informal_urdu"
    for indicator in informal_indicators:
        if indicator in text_lower:
            return "informal"
    return "formal"

def translate_with_deepl(text, target_lang):
    try:
        if not deepl_translator:
            return None, "DeepL not available"
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
        return result.text, "DeepL"
    except:
        return None, "DeepL error"

def translate_with_chatgpt(text, target_lang, source_lang="auto"):
    try:
        if not openai_client:
            return None, "ChatGPT not available"
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Keep it natural, conversational, and culturally appropriate. Only return the translated text, nothing else."},
            {"role": "user", "content": text}
        ]
        if not openai_legacy:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip(), "ChatGPT"
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip(), "ChatGPT"
    except:
        return None, "ChatGPT error"

def post_process_translation(text, source_lang, target_lang):
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])([a-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    
    if source_lang == "ur" and target_lang == "en":
        fixes = {
            "Peace be upon you": "Assalamu Alaikum",
            "Peace be upon you!": "Assalamu Alaikum!",
            "Hope you are well": "I hope you are doing well",
            "I want to know more about you": "I would like to know more about you",
            "If you find it appropriate": "If you don't mind",
            "also mention how you know me": "please also tell me how you know me",
            "how our first conversation took place": "how our first conversation started",
            "It would be a pleasure to speak with you further": "I would be happy to talk with you further",
            "official name": "officially known as",
            "consists of": "has",
            "quality education": "high-quality education",
            "making it a diverse country": "making it a diverse nation",
            "historical sites": "famous landmarks",
            "countless opportunities": "many opportunities",
            "education and employment": "education and career growth",
            "capital is": "capital city is",
            "is known for": "is famous for",
            "people from different countries": "people from many different countries",
            "the most spoken language": "the most widely spoken language",
        }
        for old, new in fixes.items():
            if old in text:
                text = text.replace(old, new)
    
    if source_lang == "en" and target_lang == "ur":
        fixes = {
            "Hello": "السلام علیکم",
            "Hello!": "السلام علیکم!",
            "Hi": "السلام علیکم",
            "Hi!": "السلام علیکم!",
            "I hope you are well": "امید ہے آپ خیریت سے ہوں گے",
            "I hope you are doing well": "امید ہے آپ خیریت سے ہوں گے",
            "I would like to know more about you": "میں آپ کے بارے میں مزید جاننا چاہتا ہوں",
            "If you don't mind": "اگر آپ کو کوئی حرج نہ ہو",
            "please also tell me how you know me": "براہ کرم مجھے یہ بھی بتائیں کہ آپ مجھے کیسے جانتے ہیں",
            "how our first conversation started": "ہماری پہلی بات چیت کیسے شروع ہوئی",
            "I would be happy to talk with you further": "آپ سے مزید بات کر کے خوشی ہوگی",
            "Statue of Liberty": "مجسمۂ آزادی",
            "famous landmarks": "تاریخی مقامات",
            "career growth": "روزگار",
            "officially known as": "جس کا سرکاری نام",
            "has": "پر مشتمل ہے",
            "high-quality education": "معیاری تعلیم",
            "diverse nation": "متنوع ملک",
            "many opportunities": "بے شمار مواقع",
        }
        for old, new in fixes.items():
            if old in text:
                text = text.replace(old, new)
    
    return text.strip()

def smart_translate(text, target_lang, source_lang=None):
    if source_lang is None:
        if any('\u0600' <= c <= '\u06FF' for c in text):
            source_lang = "ur"
        else:
            source_lang = "en"
    
    style = detect_text_style(text)
    results = {}
    
    with st.spinner("🔄 Translating..."):
        if deepl_translator:
            result, name = translate_with_deepl(text, target_lang)
            if result:
                results[name] = result
        if openai_client:
            result, name = translate_with_chatgpt(text, target_lang, source_lang)
            if result:
                results[name] = result
    
    if not results:
        return "Translation failed. Please check API keys.", "Error"
    
    selected_text = None
    selected_api = None
    
    if style in ["informal", "informal_urdu"] and "ChatGPT" in results:
        selected_text = results["ChatGPT"]
        selected_api = "ChatGPT"
    elif not selected_text and style == "formal" and "DeepL" in results:
        selected_text = results["DeepL"]
        selected_api = "DeepL"
    
    if not selected_text:
        first_api = list(results.keys())[0]
        selected_text = results[first_api]
        selected_api = first_api
    
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

st.info("🔄 **2 Translation APIs (DeepL + ChatGPT)** | Informal → ChatGPT | Formal → DeepL")

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
st.caption("🌍 LinguaBridge — Breaking Language Barriers! | 💬 Text | 🗣️ 100+ Languages | ✏️ Spell Check")
