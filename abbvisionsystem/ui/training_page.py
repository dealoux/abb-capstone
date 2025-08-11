import streamlit as st
import os
import subprocess
import json
from pathlib import Path


def training_page():
    """Training center for creating and managing vision models."""
    st.title("📊 Model Training Center")
    st.write("Train and manage object detection models")

    # Create tabs for different training functions
    tab1, tab2, tab3 = st.tabs(
        ["🎯 Object Detection Training", "📊 Training Status", "🔧 Model Management"]
    )

    with tab1:
        _object_detection_training_interface()

    with tab2:
        _training_status_interface()

    with tab3:
        _model_management_interface()


def _object_detection_training_interface():
    """Interface for training object-level defect detection."""
    st.subheader("🎯 Object Detection Training")

    st.info("Train models to detect and classify individual objects for defects")

    # Check data availability
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Data Status")

        # Check for data directory
        data_dir = "data/choco-pie"
        if os.path.exists(data_dir):
            st.success("✅ Data directory found")

            # Check subdirectories
            subdirs = ["good", "defect", "both"]
            for subdir in subdirs:
                subdir_path = os.path.join(data_dir, subdir)
                if os.path.exists(subdir_path):
                    image_count = len(
                        [
                            f
                            for f in os.listdir(subdir_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        ]
                    )
                    st.write(f"📂 {subdir}: {image_count} images")
                else:
                    st.warning(f"⚠️ {subdir} folder not found")
        else:
            st.error("❌ Data directory not found")
            st.info("💡 Expected directory structure:")
            st.code(
                """
data/choco-pie/
├── good/        # Normal samples
├── defect/      # Defective samples
└── both/        # Images with multiple objects
            """
            )

    with col2:
        st.subheader("🎛️ Training Configuration")

        # Training parameters
        epochs = st.slider("Training Epochs", 10, 100, 50)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

        # Object detection parameters
        min_area = st.slider("Min Object Area", 500, 5000, 1000)
        max_area = st.slider("Max Object Area", 10000, 100000, 50000)

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            learning_rate = st.number_input(
                "Learning Rate", 0.0001, 0.01, 0.001, format="%.4f"
            )
            use_augmentation = st.checkbox("Use Data Augmentation", value=True)
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.15)

    # Training controls
    st.markdown("---")
    col_train1, col_train2, col_train3 = st.columns(3)

    with col_train1:
        if st.button("🚀 Start Training", use_container_width=True):
            _launch_training_pipeline(epochs, batch_size, min_area, max_area)

    with col_train2:
        if st.button("📓 Open Training Notebook", use_container_width=True):
            st.info("💡 To launch the training notebook manually:")
            st.code("jupyter notebook pipelinev2.ipynb")

    with col_train3:
        if st.button("🔍 Check GPU", use_container_width=True):
            _check_gpu_status()


def _training_status_interface():
    """Training status and monitoring interface."""
    st.subheader("📊 Training Status & History")

    # Check for trained models
    models_dir = "trained_models"
    if os.path.exists(models_dir):
        st.success("✅ Trained models directory found")

        # List available models
        model_files = [
            f for f in os.listdir(models_dir) if f.endswith((".h5", ".keras", ".pb"))
        ]

        if model_files:
            st.subheader("📄 Available Models")

            for model_file in model_files:
                model_path = os.path.join(models_dir, model_file)
                file_stats = os.stat(model_path)
                file_size = file_stats.st_size / (1024 * 1024)  # MB

                col_model, col_size, col_action = st.columns([3, 1, 1])

                with col_model:
                    st.write(f"🤖 {model_file}")

                with col_size:
                    st.write(f"{file_size:.1f} MB")

                with col_action:
                    if st.button("ℹ️", key=f"info_{model_file}"):
                        _show_model_info(model_path)
        else:
            st.warning("⚠️ No trained models found")
    else:
        st.error("❌ No trained models directory found")

    # Training logs
    st.markdown("---")
    st.subheader("📋 Training Logs")

    # Check for training logs
    log_files = ["training.log", "pipeline.log"]
    found_logs = False

    for log_file in log_files:
        if os.path.exists(log_file):
            found_logs = True
            if st.button(f"📄 View {log_file}"):
                try:
                    with open(log_file, "r") as f:
                        log_content = f.read()
                    st.text_area(f"Content of {log_file}", log_content, height=300)
                except Exception as e:
                    st.error(f"Error reading {log_file}: {str(e)}")

    if not found_logs:
        st.info("ℹ️ No training logs found")


def _model_management_interface():
    """Model management interface."""
    st.subheader("🔧 Model Management")

    # Model validation
    st.subheader("✅ Model Validation")

    # Check object detection model
    object_model_path = "trained_models/object_defect_classifier.h5"
    if os.path.exists(object_model_path):
        st.success("✅ Object detection model found")

        # Validate model
        if st.button("🧪 Validate Model"):
            _validate_model(object_model_path)
    else:
        st.warning("⚠️ Object detection model not found")
        st.info("💡 Train the model using the Object Detection Training tab")

    # Model deployment
    st.markdown("---")
    st.subheader("🚀 Model Deployment")

    deployment_option = st.selectbox(
        "Deployment Target",
        ["Local Testing", "Production Server", "Edge Device", "Cloud Deployment"],
    )

    if deployment_option == "Local Testing":
        st.info("✅ Models are ready for local testing in the Object Detection page")

    elif deployment_option == "Production Server":
        st.info("🏭 Production deployment options:")
        st.write("- Copy trained_models/ directory to production server")
        st.write("- Update model paths in production configuration")
        st.write("- Test model loading and inference")

    elif deployment_option == "Edge Device":
        st.info("📱 Edge deployment considerations:")
        st.write("- Model optimization for edge devices")
        st.write("- Convert to TensorFlow Lite or ONNX format")
        st.write("- Test inference speed and accuracy")

    elif deployment_option == "Cloud Deployment":
        st.info("☁️ Cloud deployment options:")
        st.write("- Package models with application")
        st.write("- Set up cloud inference endpoints")
        st.write("- Configure auto-scaling and monitoring")

    # Model backup
    st.markdown("---")
    st.subheader("💾 Model Backup")

    if st.button("📦 Create Model Backup"):
        _create_model_backup()


def _launch_training_pipeline(epochs, batch_size, min_area, max_area):
    """Launch the training pipeline with specified parameters."""
    try:
        st.info("🚀 Launching training pipeline...")

        # Create a configuration file for the pipeline
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "min_area": min_area,
            "max_area": max_area,
            "source_dir": "data/choco-pie",
            "model_name": "object_defect_classifier",
        }

        with open("training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        st.success("✅ Configuration saved")
        st.info("💡 To start training, run:")
        st.code("jupyter notebook pipelinev2.ipynb")

        # Optionally try to run the notebook programmatically
        if st.button("🔄 Auto-run Training (Experimental)"):
            try:
                # This would require nbconvert or similar
                st.warning("⚠️ Automatic training execution not implemented yet")
                st.info("Please run the notebook manually for now")
            except Exception as e:
                st.error(f"❌ Auto-run failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to launch training: {str(e)}")


def _check_gpu_status():
    """Check GPU availability for training."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            st.success(f"✅ {len(gpus)} GPU(s) available:")
            for i, gpu in enumerate(gpus):
                st.write(f"  GPU {i}: {gpu.name}")

                # Check GPU memory
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    st.info(f"GPU {i} memory growth enabled")
                except Exception as e:
                    st.warning(f"Could not set memory growth for GPU {i}: {str(e)}")
        else:
            st.warning("⚠️ No GPUs found. Training will use CPU only.")

        # Check CUDA version
        if tf.test.is_built_with_cuda():
            st.info("✅ TensorFlow built with CUDA support")
        else:
            st.warning("⚠️ TensorFlow not built with CUDA support")

    except ImportError:
        st.error("❌ TensorFlow not installed")
    except Exception as e:
        st.error(f"❌ GPU check failed: {str(e)}")


def _validate_model(model_path):
    """Validate a trained model."""
    try:
        import tensorflow as tf

        with st.spinner("🧪 Validating model..."):
            # Load model
            model = tf.keras.models.load_model(model_path)

            # Model summary
            st.success("✅ Model loaded successfully")

            # Show model architecture
            with st.expander("🏗️ Model Architecture"):
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text("\n".join(model_summary))

            # Model metrics
            st.subheader("📊 Model Information")
            total_params = model.count_params()
            st.metric("Total Parameters", f"{total_params:,}")

            # Input/output shapes
            st.write(f"**Input Shape:** {model.input_shape}")
            st.write(f"**Output Shape:** {model.output_shape}")

            # Test prediction
            if st.button("🧪 Test Prediction"):
                # Create dummy input
                dummy_input = tf.random.normal(model.input_shape)
                prediction = model.predict(dummy_input, verbose=0)
                st.success(f"✅ Test prediction successful: {prediction.shape}")

    except Exception as e:
        st.error(f"❌ Model validation failed: {str(e)}")


def _show_model_info(model_path):
    """Show detailed model information."""
    try:
        file_stats = os.stat(model_path)

        st.subheader(f"📄 Model: {os.path.basename(model_path)}")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**File Size:** {file_stats.st_size / (1024*1024):.1f} MB")
            st.write(f"**Created:** {file_stats.st_ctime}")
            st.write(f"**Modified:** {file_stats.st_mtime}")

        with col2:
            if model_path.endswith(".h5") or model_path.endswith(".keras"):
                st.write("**Type:** Keras Model")
                st.write(
                    "**Format:** HDF5"
                    if model_path.endswith(".h5")
                    else "**Format:** Keras"
                )
            elif model_path.endswith(".pb"):
                st.write("**Type:** TensorFlow SavedModel")
                st.write("**Format:** Protocol Buffer")

    except Exception as e:
        st.error(f"❌ Error reading model info: {str(e)}")


def _create_model_backup():
    """Create a backup of trained models."""
    try:
        import shutil
        from datetime import datetime

        models_dir = "trained_models"
        if not os.path.exists(models_dir):
            st.error("❌ No models directory found")
            return

        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"model_backups/backup_{timestamp}"

        os.makedirs(backup_dir, exist_ok=True)

        # Copy all model files
        model_files = [
            f
            for f in os.listdir(models_dir)
            if f.endswith((".h5", ".keras", ".pb", ".json"))
        ]

        copied_files = 0
        for model_file in model_files:
            src = os.path.join(models_dir, model_file)
            dst = os.path.join(backup_dir, model_file)
            shutil.copy2(src, dst)
            copied_files += 1

        st.success(f"✅ Backup created: {backup_dir}")
        st.info(f"📦 {copied_files} files backed up")

    except Exception as e:
        st.error(f"❌ Backup failed: {str(e)}")
