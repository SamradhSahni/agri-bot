"""
All four baseline prompt builders in one file.
Each returns a list of messages for the NVIDIA API.
"""

# ── Agricultural examples for one-shot and few-shot ───────────────────
# These are real KCC-style pairs written manually
EXAMPLES = [
    {
        "query":  "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?",
        "state":  "उत्तर प्रदेश",
        "crop":   "मक्का",
        "intent": "कीट प्रबंधन",
        "answer": (
            "किसान भाई मक्का में फॉल आर्मी वर्म कीट के प्रबंधन के लिए "
            "इमामेक्टिन बेंज़ोइड 5% SG @ 0.4 ग्राम प्रति लीटर पानी में "
            "घोल बनाकर छिड़काव करें। छिड़काव सुबह या शाम के समय करें। "
            "15 दिन के अंतराल पर दोबारा छिड़काव करें।"
        ),
    },
    {
        "query":  "गेहूं की बुवाई का सही समय और बीज दर क्या है?",
        "state":  "हरियाणा",
        "crop":   "गेहूं",
        "intent": "फसल सलाह",
        "answer": (
            "किसान भाई गेहूं की बुवाई का सही समय नवंबर के पहले से "
            "दूसरे सप्ताह तक है। सिंचित अवस्था में बीज दर 100 किलोग्राम "
            "प्रति हेक्टेयर रखें। बीज उपचार के लिए कार्बेंडाजिम 2 ग्राम "
            "प्रति किलोग्राम बीज की दर से उपचारित करें।"
        ),
    },
    {
        "query":  "धान में झुलसा रोग का उपचार बताएं",
        "state":  "बिहार",
        "crop":   "धान",
        "intent": "रोग प्रबंधन",
        "answer": (
            "किसान भाई धान में झुलसा रोग नियंत्रण के लिए "
            "कार्बेंडाजिम 50% WP @ 1 ग्राम प्रति लीटर पानी में "
            "घोलकर छिड़काव करें। रोग की प्रारंभिक अवस्था में ही उपचार "
            "करें और 10-15 दिन के अंतराल पर दोबारा छिड़काव करें।"
        ),
    },
    {
        "query":  "सरसों में यूरिया कब और कितनी मात्रा में डालें?",
        "state":  "राजस्थान",
        "crop":   "सरसों",
        "intent": "पोषक तत्व प्रबंधन",
        "answer": (
            "किसान भाई सरसों में यूरिया की आधी मात्रा बुवाई के समय "
            "आधार खुराक के रूप में दें और शेष आधी मात्रा पहली सिंचाई "
            "के बाद टॉप ड्रेसिंग के रूप में दें। अनुशंसित मात्रा "
            "60 किलोग्राम यूरिया प्रति हेक्टेयर है।"
        ),
    },
    {
        "query":  "किसान क्रेडिट कार्ड के लिए आवेदन कैसे करें?",
        "state":  "बिहार",
        "crop":   "अन्य",
        "intent": "सरकारी योजना",
        "answer": (
            "किसान भाई किसान क्रेडिट कार्ड बनवाने के लिए अपने नजदीकी "
            "बैंक शाखा में जाएं। आवश्यक दस्तावेज — आधार कार्ड, "
            "जमीन के कागजात, पासपोर्ट साइज फोटो — साथ लेकर जाएं। "
            "आप कॉमन सर्विस सेंटर (CSC) से भी ऑनलाइन आवेदन करवा "
            "सकते हैं।"
        ),
    },
]


# ── Baseline A: Zero-shot ─────────────────────────────────────────────
def build_zero_shot(query: str, state: str, crop: str) -> list:
    """
    No system prompt. No examples.
    Just the raw question in Hindi.
    """
    return [
        {
            "role":    "user",
            "content": (
                f"राज्य: {state}\n"
                f"फसल: {crop}\n"
                f"प्रश्न: {query}"
            ),
        }
    ]


# ── Baseline B: One-shot ──────────────────────────────────────────────
def build_one_shot(query: str, state: str, crop: str) -> list:
    """
    One fixed agricultural example before the actual question.
    """
    ex = EXAMPLES[0]   # fall army worm example
    return [
        {
            "role":    "user",
            "content": (
                f"राज्य: {ex['state']}\n"
                f"फसल: {ex['crop']}\n"
                f"प्रश्न: {ex['query']}"
            ),
        },
        {
            "role":    "assistant",
            "content": ex["answer"],
        },
        {
            "role":    "user",
            "content": (
                f"राज्य: {state}\n"
                f"फसल: {crop}\n"
                f"प्रश्न: {query}"
            ),
        },
    ]


# ── Baseline C: Few-shot (5 examples) ────────────────────────────────
def build_few_shot(query: str, state: str, crop: str) -> list:
    """
    Five diverse agricultural examples covering different intents.
    """
    messages = []
    for ex in EXAMPLES:
        messages.append({
            "role":    "user",
            "content": (
                f"राज्य: {ex['state']}\n"
                f"फसल: {ex['crop']}\n"
                f"प्रश्न: {ex['query']}"
            ),
        })
        messages.append({
            "role":    "assistant",
            "content": ex["answer"],
        })

    # Actual question
    messages.append({
        "role":    "user",
        "content": (
            f"राज्य: {state}\n"
            f"फसल: {crop}\n"
            f"प्रश्न: {query}"
        ),
    })
    return messages


# ── Baseline D: Prompt-engineered ────────────────────────────────────
def build_prompt_engineered(
    query:  str,
    state:  str,
    crop:   str,
    intent: str = "कृषि सलाह",
) -> list:
    """
    Carefully designed system prompt with role, constraints,
    output format, and language enforcement.
    No examples — just an optimized instruction.
    """
    system_prompt = """आप KisanMitra AI हैं — एक विशेषज्ञ कृषि सलाहकार जो भारतीय किसानों की सहायता करते हैं।

आपके उत्तर के नियम:
1. उत्तर केवल हिंदी में दें — अंग्रेजी बिल्कुल नहीं
2. उत्तर 50-150 शब्दों में हो — न बहुत छोटा, न बहुत लंबा
3. यदि कीट या रोग की समस्या हो — दवा का नाम, मात्रा, और विधि बताएं
4. यदि फसल सलाह हो — बुवाई समय, बीज दर, खाद की मात्रा बताएं
5. यदि सरकारी योजना हो — आवेदन प्रक्रिया और संपर्क जानकारी दें
6. उत्तर व्यावहारिक और किसान के लिए उपयोगी हो
7. "किसान भाई" या "श्रीमान जी" से उत्तर शुरू करें
8. अनावश्यक जानकारी न दें — सीधे समाधान बताएं"""

    return [
        {
            "role":    "system",
            "content": system_prompt,
        },
        {
            "role":    "user",
            "content": (
                f"राज्य: {state}\n"
                f"फसल: {crop}\n"
                f"समस्या का प्रकार: {intent}\n"
                f"किसान का प्रश्न: {query}"
            ),
        },
    ]