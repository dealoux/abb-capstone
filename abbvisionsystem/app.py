import streamlit as st

# Import page modules
from abbvisionsystem.ui.detection_page import detection_page
from abbvisionsystem.ui.training_page import training_page
from abbvisionsystem.ui.camera_page import camera_page
from abbvisionsystem.ui.vision_tools_page import vision_tools_page

# Set page configuration
st.set_page_config(page_title="ABB Vision System", page_icon="♻️", layout="wide")

# Initialize session state
if "image" not in st.session_state:
    st.session_state.image = None
if "detections" not in st.session_state:
    st.session_state.detections = None
if "camera" not in st.session_state:
    st.session_state.camera = None


def main():
    """Main application entry point."""
    st.sidebar.title("🤖 ABB Vision System")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "🎯 Object Detection",
            "📷 Camera Integration",
            "🔍 Vision Tools",
            "📊 Model Training",
        ],
    )

    # Route to appropriate page
    if page == "🎯 Object Detection":
        detection_page()
    elif page == "📷 Camera Integration":
        camera_page()
    elif page == "🔍 Vision Tools":
        vision_tools_page()
    elif page == "📊 Model Training":
        training_page()


if __name__ == "__main__":
    main()
