import os
import math
import csv
import typing as t
import random
import streamlit as st

# Optional imports are wrapped so the app can run without them
try:
	from groq import Groq 
	_HAS_GROQ = True
except Exception:
	_HAS_GROQ = False

try:
	from transformers import pipeline 
	_HAS_TRANSFORMERS = True
except Exception:
	_HAS_TRANSFORMERS = False

# Allow swapping datasets and model via env vars
CSV_PATH = os.getenv("FAQ_CSV", "airline_faq.csv")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

st.set_page_config(page_title="Aurora Skies RAG Chatbot", page_icon="✈️", layout="centered")

# Data loading
@st.cache_data(show_spinner=False)
def load_faq(path: str) -> t.List[t.Dict[str, str]]:
	# Expect CSV with columns: Question, Answer
	if not os.path.exists(path):
		raise FileNotFoundError(f"Could not find FAQ CSV at {path}")
	rows: t.List[t.Dict[str, str]] = []
	with open(path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		field_map = {k.lower().strip(): k for k in reader.fieldnames or []}
		q_col = next((c for c in field_map if "question" in c), None)
		a_col = next((c for c in field_map if "answer" in c), None)
		if q_col is None or a_col is None:
			raise ValueError("CSV must contain 'Question' and 'Answer' columns")
		for r in reader:
			rows.append({
				"Question": r.get(field_map[q_col], "").strip(),
				"Answer": r.get(field_map[a_col], "").strip(),
			})
	return rows


def simple_stem(token: str) -> str:
	for suf in ["ing", "ed", "ies", "s"]:
		if token.endswith(suf) and len(token) > len(suf) + 2:
			if suf == "ies":
				return token[:-3] + "y"
			return token[: -len(suf)]
	return token


def tokenize(text: str) -> t.List[str]:
	word = []
	terms: t.List[str] = []
	for ch in text.lower():
		if ch.isalnum():
			word.append(ch)
		else:
			if word:
				terms.append(''.join(word))
				word = []
	if word:
		terms.append(''.join(word))
	stop = {"the","a","an","and","or","of","to","for","is","are","in","on","with","by","at","be","as","about"}
	base = [simple_stem(t) for t in terms if t not in stop and len(t) > 2]
	bigrams = [base[i] + "_" + base[i+1] for i in range(len(base)-1)]
	return base + bigrams

# Small hand-written synonyms to catch paraphrases
SYNONYMS: t.Dict[str, t.List[str]] = {
	"flight": ["flights", "travel", "journey", "trip", "schedule", "status"],
	"baggage": ["luggage", "bag", "suitcase", "allowance"],
	"refund": ["cancel", "cancellation", "money back", "rebook"],
	"change": ["modify", "reschedule", "rebook", "swap"],
	"delay": ["late", "postpone"],
}

# Retriever
@st.cache_resource(show_spinner=False)
def build_retriever(docs: t.List[t.Dict[str, str]]):
	# Pre-tokenize and compute idf, doc lengths, and term counts
	corpus = [(d["Question"] + " \n" + d["Answer"]).strip() for d in docs]
	tokenized = [tokenize(text) for text in corpus]
	N = len(tokenized)
	df: t.Dict[str, int] = {}
	lengths = []
	for terms in tokenized:
		lengths.append(len(terms))
		seen = set(terms)
		for tok in seen:
			df[tok] = df.get(tok, 0) + 1
	avgdl = sum(lengths)/max(1, len(lengths))
	idf = {tok: math.log(1 + (N - freq + 0.5)/(freq + 0.5)) for tok, freq in df.items()}
	tf_list: t.List[t.Dict[str, int]] = []
	for terms in tokenized:
		counts: t.Dict[str, int] = {}
		for tok in terms:
			counts[tok] = counts.get(tok, 0) + 1
		tf_list.append(counts)
	return {"idf": idf, "tf": tf_list, "lengths": lengths, "avgdl": avgdl, "docs": docs}


def expand_query_terms(terms: t.List[str]) -> t.List[str]:
	# Add synonym variants to improve matching
	expanded = list(terms)
	for t0 in terms:
		for alt in SYNONYMS.get(t0, []):
			expanded.append(simple_stem(alt))
	return expanded


def bm25_score(query_terms: t.List[str], index) -> t.List[float]:
	# Standard BM25 formulation with typical constants
	idf = index["idf"]
	tf_list = index["tf"]
	lengths = index["lengths"]
	avgdl = index["avgdl"]
	k1 = 1.5
	b = 0.75
	scores: t.List[float] = []
	for i, tf in enumerate(tf_list):
		s = 0.0
		for q in query_terms:
			if q not in idf:
				continue
			freq = tf.get(q, 0)
			if freq == 0:
				continue
			den = freq + k1 * (1 - b + b * (lengths[i] / max(1.0, avgdl)))
			s += idf[q] * (freq * (k1 + 1)) / den
		scores.append(s)
	return scores


def retrieve(query: str, index, top_k: int = 3):
	# Turn a user query into BM25 scores and return top matches
	if not query.strip():
		return []
	base_terms = tokenize(query)
	query_terms = expand_query_terms(base_terms)
	scores = bm25_score(query_terms, index)
	indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
	results = []
	for idx in indices:
		results.append({
			"question": index["docs"][idx]["Question"],
			"answer": index["docs"][idx]["Answer"],
			"score": float(scores[idx])
		})
	return results

# Generation helpers

def format_context(snippets: t.List[t.Dict[str, t.Any]]) -> str:
	# Pack retrieved FAQs into a single context block for the generator
	parts = []
	for i, s in enumerate(snippets, start=1):
		parts.append(f"[FAQ {i}] Q: {s['question']}\nA: {s['answer']}")
	return "\n\n".join(parts)


def should_abstain(snippets: t.List[t.Dict[str, t.Any]], threshold: float) -> bool:
	# Guardrail: if best score is low, prefer to abstain
	if not snippets:
		return True
	best = max(s["score"] for s in snippets)
	return best < threshold


def is_small_talk(q: str) -> bool:
	# Lightweight small‑talk detection to avoid confusing the retriever
	q = q.lower().strip()
	return q in {"hi", "hello", "hey", "hola", "yo"} or any(q.startswith(x) for x in ["hi", "hello", "hey"]) and len(q.split()) <= 3


def is_out_of_scope(q: str) -> bool:
	# Safety: block topics not suitable for an FAQ bot
	q = q.lower()
	flags = ["crash", "accident", "hijack", "bomb", "terror", "death"]
	return any(w in q for w in flags)


def generate_with_groq(prompt: str, system: str, temperature: float, max_tokens: int, top_p: float) -> str:
	client = Groq(api_key=os.getenv("GROQ_API_KEY"))
	resp = client.chat.completions.create(
		model=DEFAULT_MODEL,
		messages=[
			{"role": "system", "content": system},
			{"role": "user", "content": prompt},
		],
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
	)
	return resp.choices[0].message.content.strip()


def generate_with_transformers(prompt: str, temperature: float, max_tokens: int, top_p: float, top_k: int) -> str:
	generator = pipeline("text-generation", model="distilgpt2")
	out = generator(
		prompt,
		max_new_tokens=max_tokens,
		do_sample=True,
		temperature=temperature,
		top_p=top_p,
		top_k=top_k,
		repetition_penalty=1.05,
		pad_token_id=50256,
	)[0]["generated_text"]
	return out[len(prompt):].strip()


def extractive_answer(snippets: t.List[t.Dict[str, t.Any]], style: str) -> str:
	if not snippets:
		return "I don't know based on the provided FAQs."
	ans = snippets[0]["answer"]
	if style == "Conversational":
		prefix = random.choice(["Sure — ", "Here you go: ", "Absolutely — ", "Quick summary: "])
		return prefix + ans
	return ans


def suggest_questions(index, n: int = 3) -> t.List[str]:
	return [index["docs"][i]["Question"] for i in range(min(n, len(index["docs"])))]


def generate_answer(user_query: str, snippets: t.List[t.Dict[str, t.Any]], abstain_threshold: float, temperature: float, max_tokens: int, top_p: float, top_k: int, style: str, use_local_llm: bool, index) -> str:
	if is_small_talk(user_query):
		suggestions = " • ".join(suggest_questions(index))
		return f"Hi! I can help with airline FAQs. Try asking: {suggestions}"
	if is_out_of_scope(user_query):
		return "That topic isn’t covered in our FAQs. For emergencies, contact local authorities and airline safety teams."
	if should_abstain(snippets, abstain_threshold):
		suggestions = " • ".join(suggest_questions(index))
		return "I don't know based on the provided FAQs. Try one of these: " + suggestions

	context = format_context(snippets)
	style_hint = "Keep it friendly and natural." if style == "Conversational" else "Be concise and factual."
	system = (
		"You are a helpful airline assistant for Aurora Skies Airways. "
		"Answer ONLY using the provided FAQ context. If information is missing, say 'I don't know'. "
		+ style_hint
	)
	prompt = (
		"Context (do not reveal this section to the user):\n" + context +
		"\n\nUser question: " + user_query +
		"\n\nRules: 1) Use only the context. 2) If uncertain, reply 'I don't know'. 3) Keep answers short."
	)

	# Prefer Groq path when available; otherwise allow optional local generator
	if _HAS_GROQ and os.getenv("GROQ_API_KEY"):
		try:
			return generate_with_groq(prompt, system, temperature, max_tokens, top_p)
		except Exception:
			pass
	if use_local_llm and _HAS_TRANSFORMERS:
		try:
			return generate_with_transformers(system + "\n\n" + prompt, temperature, max_tokens, top_p, top_k)
		except Exception:
			pass
	return extractive_answer(snippets, style)


# UI

st.title("Aurora Skies Chatbot")

with st.sidebar:
	st.subheader("Settings")
	top_k = st.slider("# of FAQ passages", 1, 5, 3)
	threshold = st.slider("Abstain if similarity below", 0.0, 1.0, 0.20, 0.01)
	style = st.radio("Answer style", ["Precise", "Conversational"], index=1)
	temperature = st.slider("Temperature", 0.0, 1.5, 0.7 if style=="Conversational" else 0.4, 0.05)
	top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.92, 0.02)
	top_k_samp = st.slider("Top-k sampling", 0, 100, 40, 1)
	use_local_llm = st.checkbox("Use local generator (experimental)", value=False, help="If disabled, answers are extractive unless Groq API is set.")
	max_tokens = st.slider("Max tokens", 32, 512, 160, 16)
	st.write("Model:", DEFAULT_MODEL if os.getenv("GROQ_API_KEY") else ("distilgpt2" if (_HAS_TRANSFORMERS and use_local_llm) else "extractive-only"))
	st.markdown("Env var `GROQ_API_KEY` enables Groq-hosted open-source models.")

# Build the index once; keep app responsive with caching
try:
	docs = load_faq(CSV_PATH)
	index = build_retriever(docs)
	st.success(f"Loaded {len(docs)} FAQs from {CSV_PATH}")
except Exception as e:
	st.error(str(e))
	st.stop()

# Main
query = st.text_input("Ask a question about bookings, baggage, refunds, flight status, etc.")

if st.button("Get answer") or ("auto_run" not in st.session_state and query.strip()):
	st.session_state["auto_run"] = True
	snippets = retrieve(query, index, top_k=top_k)
	with st.expander("Retrieved FAQ context", expanded=False):
		for s in snippets:
			st.markdown(f"- Score: {s['score']:.2f}\n  - Q: {s['question']}\n  - A: {s['answer']}")

	answer = generate_answer(query, snippets, threshold, temperature, max_tokens, top_p, top_k_samp, style, use_local_llm, index)
	st.markdown("### Answer")
	st.write(answer)

	st.markdown("---")
	
