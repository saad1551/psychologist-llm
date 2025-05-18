import torch
from transformers import pipeline, StoppingCriteria
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_DIM = 384
NUM_RETRIEVED_MEMORIES = 2
MEMORY_CHUNK_SIZE = 2

# Initialize the text generation pipeline with TinyLlama
print(f"Loading text generation model: {TINYLLAMA_MODEL}...")

pipe = pipeline(
    "text-generation",
    model=TINYLLAMA_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# We use IndexFlatL2 which is a simple L2 distance index
index = faiss.IndexFlatL2(FAISS_DIM)
print(f"FAISS index initialized with dimension {FAISS_DIM}.")

# List to store the actual text of the memories, corresponding to the vectors in FAISS
memory_texts = []

# Initialize the embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

class StopOnDoubleNewline(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.double_newline_ids = tokenizer.encode("\n\n", add_special_tokens=False)
        if not self.double_newline_ids:
            raise ValueError("Tokenizer produced no tokens for '\\n\\n'. Stopping criteria cannot be used.")
        self.double_newline_ids_tensor = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.double_newline_ids_tensor is None:
            self.double_newline_ids_tensor = torch.tensor(self.double_newline_ids, device=input_ids.device, dtype=input_ids.dtype)

        for sequence_ids in input_ids:
            if len(sequence_ids) >= len(self.double_newline_ids):
                if torch.equal(sequence_ids[-len(self.double_newline_ids):], self.double_newline_ids_tensor):
                    return True
        return False
    
def add_memory(text: str):
    """Encodes text and adds it to the FAISS index and memory_texts list."""
    if not text.strip():
        return x

    # Encode the text
    # Convert to numpy array and ensure float32 dtype for FAISS
    vector = embedding_model.encode(text).astype('float32')
    # FAISS add expects a 2D array (batch_size, dimension)
    vector = np.expand_dims(vector, axis=0)

    # Add to FAISS index
    index.add(vector)

    # Store the original text
    memory_texts.append(text)
    # print(f"Added memory (vector index: {index.ntotal - 1}): '{text[:50]}...'")


def retrieve_memories(query_text: str, k: int = NUM_RETRIEVED_MEMORIES):
    """Encodes query and searches FAISS for top-k relevant memories."""
    if index.ntotal == 0:
        # print("No memories in index yet.")
        return [] # No memories to retrieve from

    # Encode the query text
    query_vector = embedding_model.encode(query_text).astype('float32')
    # FAISS search expects a 2D array
    query_vector = np.expand_dims(query_vector, axis=0)

    # Search the index
    # D is distances, I is indices of the nearest neighbors
    distances, indices = index.search(query_vector, k)

    retrieved_memory_texts = []
    # print(f"Searching memories for query: '{query_text[:50]}...'")
    for i in range(k):
        # indices is a 2D array, indices[0] gets the results for our single query
        memory_index = indices[0][i]
        distance = distances[0][i]

        # Check if the index is valid (FAISS might return -1 if k > ntotal)
        if memory_index != -1:
            # Retrieve the corresponding text
            memory_text = memory_texts[memory_index]
            retrieved_memory_texts.append(memory_text)
            # print(f"  - Retrieved memory (index {memory_index}, distance {distance:.4f}): '{memory_text[:50]}...'")


    return retrieved_memory_texts


def build_prompt(history, retrieved_memories):
    """
    Construct the prompt string from the chat history, including retrieved memories.
    """
    prompt = ""

    # Add system message first
    system_message = history[0]
    if system_message["role"] == "system":
         prompt += f"<|system|>\n{system_message['content']}\n"


    # Add retrieved memories after the system message but before chat history
    if retrieved_memories:
        prompt += "<|system|>Relevant Past Memories:\n"
        for i, memory in enumerate(retrieved_memories):
            prompt += f"- {memory}\n"
        prompt += "--- End of Memories ---\n"

    # Add recent chat history (excluding the initial system message)
    for message in history[1:]:
        if message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}\n"
        elif message["role"] == "assistant":
            prompt += f"<|assistant|>\n{message['content']}\n"
    
    # Prompt for the assistant's next response
    prompt += "<|assistant|>\n"  
    return prompt

def generate_reply(prompt_text: str, max_new_tokens: int = 128) -> str:
    """
    Generate a reply using the model with configured stopping criteria.
    Increased max_new_tokens slightly as memories add to context.
    """
    stop_criteria = StopOnDoubleNewline(pipe.tokenizer)

    outputs = pipe(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        top_k=30,
        top_p=0.8,
        stopping_criteria=[stop_criteria],
        return_full_text=False
    )
    output_text = outputs[0]["generated_text"]

    # Post-process to remove potential hallucinated follow-up turns or artifacts.
    stop_sequences = ["\n<|user|>", "\n<|system|>", "\n<|assistant|>"]

    cleaned_reply = output_text
    for seq in stop_sequences:
        if seq in cleaned_reply:
           cleaned_reply = cleaned_reply.split(seq)[0]

    # Also clean up potential trailing newlines from the stop criteria
    cleaned_reply = cleaned_reply.rstrip('\n')

    return cleaned_reply.strip()


# Manage chat history as a list of messages
# Keep the system message and few-shot examples as initial history
chat_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant role-playing as a calm and empathetic therapist. "
            "Respond directly to the user as this therapist. Your responses must be concise. "
            "Do NOT mention that you are an AI or a language model. Do NOT discuss your capabilities or limitations. "
            "Do NOT break character. Simply embody the therapist persona in your replies."
            "Use the relevant past memories provided to inform your responses if applicable, but do not explicitly refer to them as 'memories'."
        ),
    },
    # Few shot prompting - these will also be added to initial memories
    {"role": "user", "content": "Hello."},
    {"role": "assistant", "content": "Hello there. It's good to connect. How are you feeling today?"},
    {"role": "user", "content": "I've been feeling a bit anxious lately."},
    {"role": "assistant", "content": "I understand. Anxiety can be challenging. Would you like to tell me a bit more about what's been on your mind?"}
]


# Add initial chat history as memories
# We'll group them into chunks (user-assistant pairs)
print("Adding initial chat history to memory...")
for i in range(1, len(chat_history), MEMORY_CHUNK_SIZE):
    if i + 1 < len(chat_history):
        # Combine user and assistant turn into one memory chunk
        memory_text = f"User: {chat_history[i]['content']}\nAssistant: {chat_history[i+1]['content']}"
        add_memory(memory_text)
    elif i < len(chat_history):
        # Skip adding partial chunks
         pass
print("Initial memory setup complete.")

# --- Interactive Loop ---
print("\n🔹 Chat ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # User provides input. Add to history temporarily for prompt building later.
    # We add it retrieving memories based on it, but before building the final prompt.

    # Encode Current Context (user input) and Retrieve Relevant Memories
    retrieved_memories = retrieve_memories(user_input, k=NUM_RETRIEVED_MEMORIES)

    # Now add user input to the history for the prompt
    chat_history.append({"role": "user", "content": user_input})

    # Augment Prompt with Retrieved Memories and Recent History
    prompt_text = build_prompt(chat_history, retrieved_memories)
    # print("|------prompt text------|")
    # print(prompt_text)

    # Generate the assistant's response
    reply = generate_reply(prompt_text)
    print("Assistant:", reply, "\n")

    # Add assistant reply to chat history
    chat_history.append({"role": "assistant", "content": reply})

    # Encode and Store the Current Turn as a New Memory
    latest_user_turn = chat_history[-2]["content"]
    latest_assistant_turn = chat_history[-1]["content"]
    new_memory_text = f"User: {latest_user_turn}\nAssistant: {latest_assistant_turn}"
    add_memory(new_memory_text)