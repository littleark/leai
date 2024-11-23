import streamlit as st
from typing import Dict, List
import json

def metadata_collection_ui() -> Dict:
    """Collect book metadata through Streamlit UI"""
    st.subheader("📚 Book Information")
    
    with st.expander("Book Metadata", expanded=True):
        # Basic Information
        col1, col2 = st.columns(2)
        with col1:
            reading_level = st.selectbox(
                "Reading Level",
                options=["Ages 5-7", "Ages 8-10", "Ages 11-13"]
            )
            genre = st.text_input("Genre")
        
        with col2:
            setting = st.text_input("Setting")
            time_period = st.text_input("Time Period")
        
        # Themes
        st.subheader("Themes")
        themes = []
        for i in range(st.session_state.get('num_themes', 1)):
            theme = st.text_input(f"Theme {i+1}", key=f"theme_{i}")
            if theme:
                themes.append(theme)
        
        if st.button("Add Theme"):
            st.session_state.num_themes = st.session_state.get('num_themes', 1) + 1
            st.experimental_rerun()
        
        # Characters
        st.subheader("Main Characters")
        characters = []
        for i in range(st.session_state.get('num_characters', 1)):
            char_col1, char_col2 = st.columns([3, 1])
            with char_col1:
                char_name = st.text_input(f"Character {i+1}", key=f"char_{i}")
            with char_col2:
                char_role = st.selectbox(
                    "Role",
                    ["Protagonist", "Antagonist", "Supporting"],
                    key=f"char_role_{i}"
                )
            if char_name:
                characters.append({"name": char_name, "role": char_role})
        
        if st.button("Add Character"):
            st.session_state.num_characters = st.session_state.get('num_characters', 1) + 1
            st.experimental_rerun()
        
        # Discussion Points
        st.subheader("Discussion Points")
        points = []
        for i in range(st.session_state.get('num_points', 1)):
            point = st.text_area(f"Discussion Point {i+1}", key=f"point_{i}")
            if point:
                points.append(point)
        
        if st.button("Add Discussion Point"):
            st.session_state.num_points = st.session_state.get('num_points', 1) + 1
            st.experimental_rerun()
    
    # Collect all metadata
    metadata = {
        "reading_level": reading_level,
        "genre": genre,
        "setting": setting,
        "time_period": time_period,
        "themes": themes,
        "characters": characters,
        "discussion_points": points
    }
    
    return metadata

def process_uploaded_book(uploaded_file, metadata: Dict):
    """Process the uploaded book with metadata"""
    try:
        # First, process the original book content
        doc_splits = process_document(uploaded_file)
        
        # Create enhanced content documents
        enhanced_docs = []
        
        # Add metadata document
        metadata_doc = {
            "page_content": f"""
            Book Information:
            Reading Level: {metadata['reading_level']}
            Genre: {metadata['genre']}
            Setting: {metadata['setting']}
            Time Period: {metadata['time_period']}
            Themes: {', '.join(metadata['themes'])}
            """,
            "metadata": {"type": "book_info"}
        }
        enhanced_docs.append(metadata_doc)
        
        # Add character information
        for char in metadata['characters']:
            char_doc = {
                "page_content": f"""
                Character Information - {char['name']}:
                Role: {char['role']}
                Key Points:
                - Pay attention to {char['name']}'s actions and decisions
                - Notice how {char['name']} interacts with others
                - Consider {char['name']}'s development throughout the story
                """,
                "metadata": {"type": "character_info", "character": char['name']}
            }
            enhanced_docs.append(char_doc)
        
        # Add discussion points
        for point in metadata['discussion_points']:
            point_doc = {
                "page_content": f"""
                Discussion Point:
                {point}
                Think about:
                - How does this relate to the themes?
                - What evidence from the text supports your thoughts?
                - How might different characters view this?
                """,
                "metadata": {"type": "discussion_point"}
            }
            enhanced_docs.append(point_doc)
        
        # Combine original content with enhanced content
        all_docs = doc_splits + enhanced_docs
        
        return all_docs
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# Update your main app flow
def main():
    if 'metadata_collected' not in st.session_state:
        st.session_state.metadata_collected = False
    
    if not st.session_state.metadata_collected:
        metadata = metadata_collection_ui()
        if st.button("Save Metadata"):
            st.session_state.book_metadata = metadata
            st.session_state.metadata_collected = True
            st.experimental_rerun()
    
    else:
        # Show collected metadata
        st.sidebar.success("✅ Metadata Collected")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Book", type=['pdf', 'txt'])
        
        if uploaded_file is not None:
            if st.button('Process Book'):
                with st.spinner('Processing...'):
                    # Process with enhanced content
                    docs = process_uploaded_book(
                        uploaded_file, 
                        st.session_state.book_metadata
                    )
                    
                    if docs:
                        # Create vector store with enhanced docs
                        vectorstore = create_vectorstore(docs)
                        st.session_state.vectorstore = vectorstore
                        st.success('Book processed successfully!')
