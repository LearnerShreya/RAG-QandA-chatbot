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
    pass
