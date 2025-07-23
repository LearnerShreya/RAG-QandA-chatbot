"""
chat_memory.py
--------------
Sets up chat memory for contextual conversations using LangChain.
- Uses ConversationBufferMemory
- Easy to get/reset memory for RAG chain
"""

from langchain.memory import ConversationBufferMemory

# Singleton memory instance (for use in app)
_memory = None

def get_memory(memory_key: str = "chat_history", return_messages: bool = True):
    """
    Returns a ConversationBufferMemory instance (singleton).
    """
    global _memory
    if _memory is None:
        _memory = ConversationBufferMemory(memory_key=memory_key, return_messages=return_messages)
    return _memory

def reset_memory():
    """
    Resets the conversation memory.
    """
    global _memory
    _memory = None

if __name__ == "__main__":
    # Example usage
    mem = get_memory()
    mem.save_context({"input": "What is a home loan?"}, {"output": "A home loan is a loan taken to buy a house."})
    mem.save_context({"input": "Is credit history important?"}, {"output": "Yes, it affects loan approval."})
    print("Chat history:")
    for msg in mem.chat_memory.messages:
        print(msg)
    reset_memory()
    print("Memory reset. New instance:", get_memory())
