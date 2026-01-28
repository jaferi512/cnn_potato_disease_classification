🥔 Potato Disease Classification using ResNet
A deep learning project utilizing Residual Networks (ResNet) to identify and classify diseases in potato leaves. This tool aims to assist in early-stage crop pathology detection.


🔍 Overview
This project addresses the challenge of agricultural loss due to Early Blight and Late Blight. By leveraging a ResNet architecture, we utilize "skip connections" to train deeper networks without the vanishing gradient problem, resulting in high-accuracy classification.

🏗 System Architecture
The model pipeline follows these standard stages:

Preprocessing: Resizing images to 224×224 and normalizing pixel values.

Data Augmentation: Random rotations, flips, and zooms to prevent overfitting.

Feature Extraction: Using pre-trained ResNet layers.

Classification: A Fully Connected (FC) layer with Softmax activation.

⚙️ Installation & Setup
1. Clone the Repository

Bash
git clone https://github.com/yourusername/potato-disease-resnet.git
cd potato-disease-resnet
2. Environment Setup

<details> <summary><b>Click to expand: macOS / Linux instructions</b></summary>

Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
</details>

<details> <summary><b>Click to expand: Windows instructions</b></summary>

Bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
</details>

📂 Dataset Structure
Place your raw images in the potatoData/ directory. The notebook is configured to read from subfolders as labels:

Plaintext
potatoData/
├── Healthy/        # Images of healthy leaves
├── Early_Blight/   # Images of Alternaria solani
└── Late_Blight/    # Images of Phytophthora infestans
🚀 Usage
Ensure your virtual environment is active.

Launch the notebook: jupyter notebook.

Open the .ipynb file and run the cells.

The notebook will automatically save the best-performing model as .h5 or .pth.

📊 Results
Upon completion, the model generates:

Accuracy: Typically >95% on the validation set.

Confusion Matrix: To visualize which classes are being misidentified.

Inference: A sample script to test the model on individual images.

🤝 Contributing
Fork the Project.

Create your Feature Branch (git checkout -b feature/AmazingFeature).

Commit your Changes (git commit -m 'Add some AmazingFeature').

Push to the Branch (git push origin feature/AmazingFeature).
