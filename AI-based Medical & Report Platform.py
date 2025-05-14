import cv2
import numpy as np
import pydicom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 1. Image Preprocessing
def preprocess_image(image_path):
    if image_path.endswith(".dcm"):  # DICOM files
        dicom_data = pydicom.dcmread(image_path)
        image = dicom_data.pixel_array
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image for model input
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    image = np.expand_dims(image, axis=-1)  # Adding channel dimension (for grayscale)
    image = image / 255.0  # Normalize
    return image

# 2. Create Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # Example for binary classification (healthy vs disease)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Training the Model (Using Sample Dataset)
def train_model(model, train_dir, val_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    model.fit(train_generator, epochs=10, validation_data=validation_generator)

# 4. Report Generation
def generate_report(disease_status, image_path, report_file):
    c = canvas.Canvas(report_file, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Add basic info
    c.drawString(100, 750, f"Medical Imaging Report for: {image_path}")
    c.drawString(100, 730, f"Disease Detected: {disease_status}")
    c.drawString(100, 710, "Additional details:")
    c.drawString(100, 690, "The analysis indicates a potential presence of abnormality in the scanned image.")

    # Save the PDF report
    c.save()

# 5. Analyze Image and Generate Report
def analyze_image_and_generate_report(image_path, model, report_file):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Model inference (predict disease)
    prediction = model.predict(image)
    disease_status = "Positive for Disease" if np.argmax(prediction) == 1 else "Healthy"

    # Generate the report
    generate_report(disease_status, image_path, report_file)

# 6. Main Function to Execute Everything
if __name__ == "__main__":
    # Example to create the model (for training and inference)
    model = create_model()

    # Optional: Uncomment this if you want to train the model
    # train_model(model, 'train_data/', 'val_data/')  # Path to your dataset

    # Test image analysis and report generation
    analyze_image_and_generate_report("test_xray.png", model, "final_report.pdf")
