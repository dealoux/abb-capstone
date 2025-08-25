import os
import streamlit as st
import subprocess
import json
from pathlib import Path


def training_center_page():
    """Training center for creating and managing vision models."""
    st.title("📊 Vision Training Center")
    st.write("Train and manage computer vision models for defect detection")

    # Show current model status
    _show_model_status()

    # Training pipeline selection
    training_type = st.selectbox(
        "Select Training Pipeline",
        [
            "🤖 Complete Pipeline (YOLO + ResNet)",
            "🎯 YOLO Multi-object Detection",
            "🧠 ResNet Classification",
            "📁 Dataset Management",
            "📈 Model Performance",
        ],
    )

    if training_type == "🤖 Complete Pipeline (YOLO + ResNet)":
        complete_pipeline_interface()
    elif training_type == "🎯 YOLO Multi-object Detection":
        yolo_training_interface()
    elif training_type == "🧠 ResNet Classification":
        resnet_training_interface()
    elif training_type == "📁 Dataset Management":
        dataset_management_interface()
    elif training_type == "📈 Model Performance":
        model_performance_interface()


def _show_model_status():
    """Show current trained model status"""
    st.subheader("🔍 Current Model Status")

    model_base_path = "trained_models"

    # Check for YOLO model
    yolo_paths = [
        os.path.join(model_base_path, "yolo_defect_detector", "weights", "best.pt"),
        os.path.join(model_base_path, "yolo_defect_detector", "weights", "last.pt"),
        os.path.join(model_base_path, "best.pt"),
    ]

    yolo_model_found = False
    yolo_model_path = None
    for path in yolo_paths:
        if os.path.exists(path):
            yolo_model_found = True
            yolo_model_path = path
            break

    # Check for ResNet model
    resnet_paths = [
        os.path.join(model_base_path, "resnet_defect_classifier.keras"),
        os.path.join(model_base_path, "resnet_defect_classifier.h5"),
    ]

    resnet_model_found = False
    resnet_model_path = None
    for path in resnet_paths:
        if os.path.exists(path):
            resnet_model_found = True
            resnet_model_path = path
            break

    # Display status
    col1, col2 = st.columns(2)

    with col1:
        if yolo_model_found:
            st.success("✅ YOLO Model Available")
            st.info(f"📍 Path: {yolo_model_path}")
            if os.path.exists(yolo_model_path):
                # Get file size
                size_mb = os.path.getsize(yolo_model_path) / (1024 * 1024)
                st.info(f"📊 Size: {size_mb:.1f} MB")
        else:
            st.warning("⚠️ No YOLO model found")
            st.info("💡 Train a model to enable detection")

    with col2:
        if resnet_model_found:
            st.success("✅ ResNet Model Available")
            st.info(f"📍 Path: {resnet_model_path}")
            if os.path.exists(resnet_model_path):
                # Get file size
                size_mb = os.path.getsize(resnet_model_path) / (1024 * 1024)
                st.info(f"📊 Size: {size_mb:.1f} MB")
        else:
            st.warning("⚠️ No ResNet model found")
            st.info("💡 Train a model to enable classification")


def complete_pipeline_interface():
    """Interface for running the complete training pipeline."""
    st.header("🤖 Complete Training Pipeline")
    st.write("Train both YOLO and ResNet models for comprehensive defect detection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Dataset Configuration")

        # Data source selection
        data_source = st.text_input(
            "Data Directory Path",
            value="data/choco-pie",
            help="Path to your dataset with 'good' and 'defect' folders",
        )

        # Check if data exists
        if os.path.exists(data_source):
            good_dir = os.path.join(data_source, "good")
            defect_dir = os.path.join(data_source, "defect")

            if os.path.exists(good_dir) and os.path.exists(defect_dir):
                good_count = len(
                    [
                        f
                        for f in os.listdir(good_dir)
                        if f.endswith((".jpg", ".jpeg", ".png", ".bmp", ".JPG"))
                    ]
                )
                defect_count = len(
                    [
                        f
                        for f in os.listdir(defect_dir)
                        if f.endswith((".jpg", ".jpeg", ".png", ".bmp", ".JPG"))
                    ]
                )

                st.success(f"✅ Dataset found!")
                st.info(f"📊 Normal samples: {good_count}")
                st.info(f"📊 Defect samples: {defect_count}")

                data_valid = True
            else:
                st.error(
                    "❌ Dataset structure invalid. Need 'good' and 'defect' folders."
                )
                data_valid = False
        else:
            st.warning("⚠️ Dataset path not found")
            data_valid = False

    with col2:
        st.subheader("⚙️ Training Configuration")

        # Training options
        train_yolo = st.checkbox(
            "Train YOLO Model", value=True, help="For multi-object detection"
        )
        train_resnet = st.checkbox(
            "Train ResNet Model", value=True, help="For single-object classification"
        )

        # Training parameters
        st.write("**Training Parameters:**")
        yolo_epochs = st.slider("YOLO Epochs", 10, 200, 50)
        resnet_epochs = st.slider("ResNet Epochs", 10, 100, 25)

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            batch_size = st.slider("Batch Size", 8, 32, 16)
            image_size = st.slider("Image Size", 416, 1024, 640)
            train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.7)
            val_ratio = st.slider("Validation Ratio", 0.1, 0.3, 0.15)

    # Training execution
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🚀 Start Training", disabled=not data_valid, use_container_width=True
        ):
            _run_complete_training_pipeline(
                data_source,
                train_yolo,
                train_resnet,
                yolo_epochs,
                resnet_epochs,
                batch_size,
                image_size,
            )

    with col2:
        if st.button("🔍 Test Setup", use_container_width=True):
            _test_training_setup()

    with col3:
        if st.button("📊 View Progress", use_container_width=True):
            _show_training_progress()


def _run_complete_training_pipeline(
    data_source,
    train_yolo,
    train_resnet,
    yolo_epochs,
    resnet_epochs,
    batch_size,
    image_size,
):
    """Execute the complete training pipeline"""

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    try:
        with log_container:
            with st.expander("📝 Training Logs", expanded=True):
                log_placeholder = st.empty()

                status_text.text("🔄 Initializing training pipeline...")
                progress_bar.progress(10)

                # Import and run the pipeline
                try:
                    # Import the pipeline function
                    import sys

                    sys.path.append(os.path.join(os.getcwd()))

                    from pipelinev2 import run_complete_pipeline, test_pipeline_setup

                    # Test setup first
                    status_text.text("🔍 Testing pipeline setup...")
                    if not test_pipeline_setup():
                        st.error("❌ Pipeline setup failed. Check dependencies.")
                        return

                    progress_bar.progress(20)

                    # Run the training pipeline
                    status_text.text("🚀 Starting training pipeline...")

                    results = run_complete_pipeline(
                        source_data_dir=data_source,
                        use_yolo=train_yolo,
                        use_classification=train_resnet,
                        train_yolo_epochs=yolo_epochs,
                        train_classification_epochs=resnet_epochs,
                    )

                    progress_bar.progress(100)
                    status_text.text("✅ Training completed successfully!")

                    # Display results
                    if results:
                        st.success("🎉 Training completed successfully!")
                        _display_training_results(results)

                        # Show where models are saved
                        st.info("📁 **Models saved to:**")
                        if train_yolo:
                            st.info(
                                "🎯 YOLO: `trained_models/yolo_defect_detector/weights/best.pt`"
                            )
                        if train_resnet:
                            st.info(
                                "🧠 ResNet: `trained_models/resnet_defect_classifier.keras`"
                            )

                        # Refresh model status
                        st.rerun()

                except ImportError as e:
                    st.error(f"❌ Import error: {e}")
                    st.info("💡 Make sure pipelinev2.py is in the correct location")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    st.info("💡 Check the logs above for detailed error information")

    except Exception as e:
        st.error(f"❌ Pipeline execution failed: {e}")
        status_text.text("❌ Training failed")


def _test_training_setup():
    """Test if training setup is correct"""
    st.info("🔍 Testing training setup...")

    try:
        # Test imports
        test_results = []

        try:
            from abbvisionsystem.training_pipeline.data_manager import organize_dataset

            test_results.append(("✅", "data_manager import"))
        except ImportError as e:
            test_results.append(("❌", f"data_manager import: {e}"))

        try:
            from abbvisionsystem.training_pipeline.yolov8_trainer import (
                YOLODefectDetector,
            )

            test_results.append(("✅", "yolo_trainer import"))
        except ImportError as e:
            test_results.append(("❌", f"yolo_trainer import: {e}"))

        try:
            from abbvisionsystem.training_pipeline.resnet_trainer import (
                DefectClassificationModel,
            )

            test_results.append(("✅", "resnet_trainer import"))
        except ImportError as e:
            test_results.append(("❌", f"resnet_trainer import: {e}"))

        try:
            from ultralytics import YOLO

            test_results.append(("✅", "ultralytics available"))
        except ImportError:
            test_results.append(
                ("⚠️", "ultralytics not installed (pip install ultralytics)")
            )

        try:
            import tensorflow as tf

            test_results.append(("✅", f"tensorflow {tf.__version__}"))
        except ImportError:
            test_results.append(("❌", "tensorflow not installed"))

        # Display results
        for status, message in test_results:
            if status == "✅":
                st.success(f"{status} {message}")
            elif status == "⚠️":
                st.warning(f"{status} {message}")
            else:
                st.error(f"{status} {message}")

    except Exception as e:
        st.error(f"Setup test failed: {e}")


def _display_training_results(results):
    """Display training results in a formatted way"""
    st.subheader("📈 Training Results")

    if "yolo" in results and results["yolo"]:
        st.write("**🎯 YOLO Model Results:**")
        yolo_results = results["yolo"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{yolo_results['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{yolo_results['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{yolo_results['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{yolo_results['f1_score']:.3f}")

    if "classification" in results and results["classification"]:
        st.write("**🧠 ResNet Model Results:**")
        class_results = results["classification"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{class_results['test_accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{class_results['test_precision']:.3f}")
        with col3:
            st.metric("Recall", f"{class_results['test_recall']:.3f}")
        with col4:
            # Calculate F1
            p = class_results["test_precision"]
            r = class_results["test_recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            st.metric("F1 Score", f"{f1:.3f}")


def _show_training_progress():
    """Show current training progress if available"""
    st.info("📊 Training progress monitoring would be implemented here")
    st.write("This would show:")
    st.write("- Real-time training metrics")
    st.write("- Loss curves")
    st.write("- Validation performance")
    st.write("- ETA and current epoch")


def yolo_training_interface():
    """YOLO-specific training interface"""
    st.header("🎯 YOLO Multi-object Detection Training")
    st.info("🚧 YOLO-specific training interface - Coming soon!")


def resnet_training_interface():
    """ResNet-specific training interface"""
    st.header("🧠 ResNet Classification Training")
    st.info("🚧 ResNet-specific training interface - Coming soon!")


def dataset_management_interface():
    """Dataset management interface"""
    st.header("📁 Dataset Management")
    st.info("🚧 Dataset management interface - Coming soon!")


def model_performance_interface():
    """Model performance analysis interface"""
    st.header("📈 Model Performance Analysis")

    # Model selection
    model_dir = "trained_models"

    if os.path.exists(model_dir):
        # List available models
        model_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith((".h5", ".keras", ".pt")):
                    rel_path = os.path.relpath(os.path.join(root, file), model_dir)
                    model_files.append(rel_path)

        if model_files:
            selected_model = st.selectbox("Select Model to Analyze", model_files)

            st.info(f"📊 Performance analysis for: {selected_model}")
            st.write("This would show:")
            st.write("- Confusion matrix")
            st.write("- ROC curves")
            st.write("- Precision-recall curves")
            st.write("- Sample predictions")
            st.write("- Model size and inference speed")
        else:
            st.warning("⚠️ No trained models found")
    else:
        st.warning("⚠️ Models directory not found")
