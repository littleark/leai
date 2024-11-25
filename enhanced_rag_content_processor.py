import json
from dataclasses import dataclass
from typing import List, Optional
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class BookMetadata:
    title: str
    author: str
    reading_level: str
    themes: List[str]
    main_characters: List[str]
    setting: str

@dataclass
class EnhancedContent:
    chapter: str
    content: str
    key_events: List[str]
    vocabulary: List[dict]
    discussion_points: List[str]

def create_enhanced_rag_content(book_text: str, metadata: BookMetadata) -> List[dict]:
    """Create enhanced RAG content with multiple content types"""
    documents = []

    # 1. Add book metadata as a separate document
    metadata_doc = {
        "content": f"""
        {metadata.title}

        Book Information:
        Title: {metadata.title}
        Author: {metadata.author}
        Reading Level: {metadata.reading_level}
        Themes: {', '.join(metadata.themes)}
        Main Characters: {', '.join(metadata.main_characters)}
        Setting: {metadata.setting}
        """,
        "type": "metadata",
        "metadata": {"section": "book_info"}
    }
    documents.append(metadata_doc)

    # 2. Add pre-written discussion prompts and questions
    discussion_templates = [
        {
            "content": f"Discussion question about character {char}: What do you think about {char}'s actions when...",
            "type": "discussion_prompt",
            "metadata": {"character": char}
        }
        for char in metadata.main_characters
    ]
    documents.extend(discussion_templates)

    # 3. Add reading comprehension guidance
    comprehension_doc = {
        "content": f"""
        Key points to remember while reading {metadata.title}:
        - Pay attention to how the characters change
        - Notice the important events in each chapter
        - Think about the main themes: {', '.join(metadata.themes)}
        - Consider how the setting affects the story
        """,
        "type": "reading_guidance",
        "metadata": {"section": "comprehension"}
    }
    documents.append(comprehension_doc)

    # 4. Add character analysis templates
    for character in metadata.main_characters:
        char_doc = {
            "content": f"""
            Character Analysis - {character}:
            - Think about their personality traits
            - Notice how they interact with others
            - Watch how they handle challenges
            - Consider how they change throughout the story
            """,
            "type": "character_analysis",
            "metadata": {"character": character}
        }
        documents.append(char_doc)

    # 5. Add theme exploration prompts
    for theme in metadata.themes:
        theme_doc = {
            "content": f"""
            Theme Exploration - {theme}:
            - Look for examples of this theme in the story
            - Think about how different characters relate to this theme
            - Consider what the author might be saying about {theme}
            """,
            "type": "theme_analysis",
            "metadata": {"theme": theme}
        }
        documents.append(theme_doc)

    # 6. Process original book text with enhanced context
    # chapters = split_into_chapters(book_text)
    # for chapter_num, chapter_text in enumerate(chapters, 1):
    #     chapter_doc = {
    #         "content": chapter_text,
    #         "type": "chapter_content",
    #         "metadata": {
    #             "chapter": chapter_num,
    #             "section": "main_text"
    #         }
    #     }
    #     documents.append(chapter_doc)
    #

    # templates that encourage kids to imagine alternative endings or write a new scene
    documents.append({
        "content": f"If you could change the ending of {metadata.title}, what would you make happen? Why?",
        "type": "creative_prompt",
        "metadata": {"focus": "story_alteration"}
    })
    for character in metadata.main_characters:
        char_doc = {
            "content": f"Imagine {character} meets a new character in the jungle. Who would it be, and how would they interact?",
            "type": "creative_prompt",
            "metadata": {"character": character}
        }
        documents.append(char_doc)

    # questions that help kids think critically about the ethical choices characters face
    for character in metadata.main_characters:
        char_doc = {
            "content": f"What do you think {character} should have done when they faced [specific situation]? Was it the right choice?",
            "type": "moral_dilemma",
            "metadata": {"character": character}
        }
        documents.append(char_doc)

    # background on the setting or the time period in which The Jungle Book was written
    documents.append({
        "content": f"Did you know that {metadata.title} is set in colonial India? What do you notice about how the animals and people interact? How do you think the time period influences the story?",
        "type": "contextual_info",
        "metadata": {"focus": "historical_context"}
    })

    # mini-challenges or quizzes for kids
    documents.append({
        "content": f"Quiz: Can you list three themes from {metadata.title}? Bonus: Give an example from the book for each theme.",
        "type": "challenge",
        "metadata": {"focus": "quiz"}
    })

    # Encourage kids to see the story from different perspectives
    for character in metadata.main_characters:
        char_doc = {
            "content": f"Imagine you are {character}. How would you feel when [specific event] happens?",
            "type": "empathy_prompt",
            "metadata": {"character": character}
        }
        documents.append(char_doc)

    # Help kids relate the themes to their own lives
    for theme in metadata.themes:
        theme_doc = {
            "content": f"The theme of {theme} is important in {metadata.title}. Can you think of a time when you experienced or noticed {theme} in real life?",
                "type": "real_world_connection",
                "metadata": {"theme": theme}
        }
        documents.append(theme_doc)

    # {
    #     "content": f"Can you find three interesting words in Chapter {chapter}? What do they mean, and why do you think the author used them?",
    #     "type": "vocabulary",
    #     "metadata": {"chapter": chapter}
    # }
    documents.append({
        "content": f"Can you find three interesting words you read in {metadata.title}? What do they mean, and why do you think the author used them?",
        "type": "vocabulary",
        "metadata": {"focus": "vocabulary"}
    })

    return documents

def process_document_with_enhancements(uploaded_file, chunk_size=800, chunk_overlap=100):
    """Enhanced document processor with metadata and additional content"""
    # Read the original book content
    book_text = uploaded_file.getvalue().decode()

    # Extract or provide metadata (could be from a separate upload or UI input)
    # metadata = BookMetadata(
    #     title=extract_title(book_text),
    #     author=extract_author(book_text),
    #     reading_level="ages 8-12",  # Could be determined or input
    #     themes=extract_themes(book_text),
    #     main_characters=extract_characters(book_text),
    #     setting=extract_setting(book_text)
    # )

    metadata = BookMetadata(
        title="The Jungle Book",
        author= "Rudyard Kipling",
        reading_level= "Intermediate to Advanced",
        themes= [
            "Nature and Wilderness",
            "Identity and Belonging",
            "Friendship and Loyalty",
            "Survival and Adventure",
            "Man vs. Nature"
        ],
        main_characters = [
            "Mowgli",
            "Baloo (the bear)",
            "Bagheera (the black panther)",
            "Shere Khan (the tiger)",
            "Kaa (the python)",
            "Akela (the wolf pack leader)",
            "Rikki-Tikki-Tavi (the mongoose)"
        ],
        setting = "The jungles of India during the late 19th century")

    # Create enhanced content
    enhanced_docs = create_enhanced_rag_content(book_text, metadata)

    # Process with text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True
    )

    # Split while preserving metadata
    processed_docs = []
    for doc in enhanced_docs:
        splits = text_splitter.create_documents(
            [doc["content"]],
            metadatas=[doc["metadata"]] * (len(doc["content"]) // chunk_size + 1)
        )
        processed_docs.extend(splits)

    return processed_docs

# Helper functions (to be implemented based on your needs)
# def split_into_chapters(text: str) -> List[str]:
#     """Split book text into chapters"""
#     # Implementation depends on book format
#     pass

# def extract_title(text: str) -> str:
#     """Extract book title from text"""
#     # Implementation depends on book format
#     pass

# def extract_author(text: str) -> str:
#     """Extract author from text"""
#     # Implementation depends on book format
#     pass

# def extract_themes(text: str) -> List[str]:
#     """Extract main themes from text"""
#     # Could use LLM for this
#     pass

# def extract_characters(text: str) -> List[str]:
#     """Extract main characters from text"""
#     # Could use NER or LLM for this
#     pass

# def extract_setting(text: str) -> str:
#     """Extract story setting from text"""
#     # Could use LLM for this
#     pass
