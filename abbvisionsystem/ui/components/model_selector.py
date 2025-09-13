"""Model selection component for the UI."""

import streamlit as st
import os
from typing import Optional
from abbvisionsystem.models.model_factory import ModelFactory
import logging

logger = logging.getLogger(__name__)


class ModelSelector:
    """Component for handling model selection in the UI."""

    @staticmethod
    def render_model_selection(model_type: str) -> Optional[str]:
        """Render model selection interface and return selected model path."""
        if model_type != "defect_yolo":
            return None

        st.subheader("🤖 YOLO Model Selection")

        # Get available models from factory
        available_models = ModelFactory.get_available_models(model_type)

        # Current model status
        ModelSelector._show_current_model_status()

        # Trained models section
        trained_models = available_models.get("trained", [])
        if trained_models:
            selected_path = ModelSelector._render_trained_models(trained_models)
            if selected_path:
                return selected_path

        # Pretrained models section
        pretrained_models = available_models.get("pretrained", [])
        if pretrained_models:
            selected_path = ModelSelector._render_pretrained_models(pretrained_models)
            if selected_path:
                return selected_path

        # Upload section
        uploaded_path = ModelSelector._render_upload_section(model_type)
        if uploaded_path:
            return uploaded_path

        # Clear model option
        ModelSelector._render_clear_option()

        # Return currently selected model path
        return getattr(st.session_state, "selected_model_path", None)

    @staticmethod
    def _show_current_model_status():
        """Show current model status."""
        if hasattr(st.session_state, "selected_model_path"):
            current_model = st.session_state.selected_model_path
            if os.path.exists(current_model):
                st.success("✅ Model Loaded")
                model_name = os.path.basename(current_model)
                st.info(f"📁 Current: {model_name}")

                model_size_mb = os.path.getsize(current_model) / (1024 * 1024)
                st.info(f"📦 Size: {model_size_mb:.1f} MB")
            else:
                st.warning(f"⚠️ Model file not found: {os.path.basename(current_model)}")
        else:
            st.info("No model selected - will use auto-discovery")

    @staticmethod
    def _render_trained_models(trained_models) -> Optional[str]:
        """Render trained models selection."""
        st.write("**🎯 Trained Models (Recommended)**")

        trained_options = ["None"] + [
            f"{m['name']} ({m['size_mb']:.1f}MB)" for m in trained_models
        ]
        trained_paths = [None] + [m["path"] for m in trained_models]

        selected_trained_idx = 0
        if hasattr(st.session_state, "selected_model_path"):
            current_path = st.session_state.selected_model_path
            if current_path in trained_paths:
                selected_trained_idx = trained_paths.index(current_path)

        selected_trained = st.selectbox(
            "Choose trained model:",
            options=trained_options,
            index=selected_trained_idx,
            key="trained_model_selector",
        )

        if selected_trained != "None":
            selected_path = trained_paths[trained_options.index(selected_trained)]
            model_info = next(m for m in trained_models if m["path"] == selected_path)

            # Show model details
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Modified:** {model_info['date_str']}")
                st.write(f"**Type:** {model_info['model_type']}")
            with col2:
                st.write(f"**Accuracy:** {model_info.get('accuracy', 'Unknown')}")
                st.write(f"**Epochs:** {model_info.get('epochs', 'Unknown')}")

            if st.button("🔄 Load Trained Model"):
                st.session_state.selected_model_path = selected_path
                st.success(f"✅ Loaded: {model_info['name']}")
                st.rerun()
                return selected_path

        return None

    @staticmethod
    def _render_pretrained_models(pretrained_models) -> Optional[str]:
        """Render pretrained models selection."""
        st.write("---")
        st.write("**⚡ Pretrained Models (General Purpose)**")

        pretrained_options = ["None"] + [
            f"{m['name']} - {m['description']}" for m in pretrained_models
        ]
        pretrained_paths = [None] + [m["path"] for m in pretrained_models]

        selected_pretrained_idx = 0
        if hasattr(st.session_state, "selected_model_path"):
            current_path = st.session_state.selected_model_path
            if current_path in pretrained_paths:
                selected_pretrained_idx = pretrained_paths.index(current_path)

        selected_pretrained = st.selectbox(
            "Choose pretrained model:",
            options=pretrained_options,
            index=selected_pretrained_idx,
            key="pretrained_model_selector",
        )

        if selected_pretrained != "None":
            selected_path = pretrained_paths[
                pretrained_options.index(selected_pretrained)
            ]
            model_info = next(
                m for m in pretrained_models if m["path"] == selected_path
            )

            st.info(f"ℹ️ {model_info['description']}")
            st.warning("⚠️ Pretrained models may not be optimized for defect detection")

            if st.button("🔄 Load Pretrained Model"):
                st.session_state.selected_model_path = selected_path
                st.success(f"✅ Loaded: {model_info['name']}")
                st.rerun()
                return selected_path

        return None

    @staticmethod
    def _render_upload_section(model_type: str) -> Optional[str]:
        """Render model upload section."""
        st.write("---")
        st.write("**📤 Upload Custom Model**")

        uploaded_model = st.file_uploader(
            "Upload YOLO model (.pt file)",
            type=["pt"],
            key="model_upload",
            help="Upload a custom trained YOLO model file",
        )

        if uploaded_model is not None:
            try:
                # Save uploaded model
                model_dir = "uploaded_models"
                os.makedirs(model_dir, exist_ok=True)

                model_path = os.path.join(model_dir, uploaded_model.name)

                with open(model_path, "wb") as f:
                    f.write(uploaded_model.read())

                # Validate model
                if ModelFactory.validate_model(model_type, model_path):
                    st.session_state.selected_model_path = model_path
                    st.success(f"✅ Uploaded and loaded: {uploaded_model.name}")

                    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    st.info(f"📦 Size: {model_size_mb:.1f} MB")

                    st.rerun()
                    return model_path
                else:
                    st.error("Invalid YOLO model file")
                    os.remove(model_path)

            except Exception as e:
                st.error(f"Error uploading model: {str(e)}")

        return None

    @staticmethod
    def _render_clear_option():
        """Render clear model option."""
        if st.button("🗑️ Clear Selected Model"):
            if hasattr(st.session_state, "selected_model_path"):
                del st.session_state.selected_model_path
            st.success("Model selection cleared - will use auto-discovery")
            st.rerun()
