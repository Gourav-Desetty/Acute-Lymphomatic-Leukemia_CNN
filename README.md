<h1 align="center">Acute Lymphoblastic Leukemia Detection using Deep Learning & LLM</h1>

<p align="center">
A complete end-to-end AI system for classifying microscopic blood smear images into 
<b>ALL (cancerous)</b> and <b>HEM (healthy)</b> and generating automated pathology-style reports.
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This project integrates <b>Deep Convolutional Neural Networks (CNNs)</b> with 
<b>Large Language Models (LLMs)</b> to perform two tasks:
</p>

<ul>
  <li><b>Classify leukemia from blood smear images</b> using a DenseNet-121 backbone.</li>
  <li><b>Generate clinical-style diagnostic reports</b> based on the model's predictions.</li>
</ul>

<p>
The goal is to accelerate leukemia screening by combining the power of medical imaging and AI-driven reporting.
</p>

<hr>

<h2>📊 Dataset Summary</h2>

<table>
  <tr><td><b>Total Images</b></td><td>10,661</td></tr>
  <tr><td><b>Cancer (ALL)</b></td><td>7,272</td></tr>
  <tr><td><b>Healthy (HEM)</b></td><td>3,389</td></tr>
  <tr><td><b>Unique Subjects</b></td><td>892</td></tr>
  <tr><td><b>Avg. Images per Subject</b></td><td>≈ 12</td></tr>
</table>

<p>
Subject-wise splitting was used to prevent data leakage and ensure realistic performance evaluation.
</p>

<hr>

<h2>🚀 Key Features</h2>

<ul>
  <li>📁 Fully modular and production-grade project structure</li>
  <li>🧠 CNN-based leukemia classifier using DenseNet-121</li>
  <li>📝 Automated diagnostic report generation with LLMs</li>
  <li>📈 Visualization of loss, accuracy, F1-score, ROC-AUC</li>
  <li>🧪 Clean training/evaluation pipeline with early stopping</li>
</ul>

<hr>

<h2>🏗 Project Architecture</h2>

<pre>
leukemia/
│
├── constant/
├── data/
│   ├── ingestion.py
│   ├── transformation.py
│   └── validation.py
│
├── entity/
│   ├── artifact_entity.py
│   └── model_architecture.py
│
├── inference/
│   └── predict.py
│
├── nlp/
│   ├── chains/
│   ├── generator/
│   ├── prompts/
│   └── loaders/
│
├── utils/
└── main.py
└── requirements.txt
</pre>

<p>
This architecture follows industry standards: clean separation of data, model, inference, logging, and NLP components.
</p>

<hr>

<h2>📈 Model Performance</h2>

<table>
  <tr><td><b>Validation Accuracy</b></td><td>92.52%</td></tr>
  <tr><td><b>Validation F1 Score</b></td><td>93.24%</td></tr>
  <tr><td><b>Validation Recall</b></td><td>92.52%</td></tr>
  <tr><td><b>Validation ROC-AUC</b></td><td>0.9785</td></tr>
</table>

<p>
The model converged by Epoch 12, with early stopping to avoid overfitting.
</p>

<hr>

<h2>🧬 End-to-End Pipeline</h2>

<h3>1️⃣ Data Processing</h3>
<ul>
  <li>Normalization & resizing</li>
  <li>Folder-wise extraction</li>
  <li>Subject-wise stratified split</li>
</ul>

<h3>2️⃣ Model Training</h3>
<ul>
  <li>DenseNet-121 backbone</li>
  <li>Adam optimizer</li>
  <li>Weighted loss for class imbalance</li>
  <li>Learning rate scheduling</li>
</ul>

<h3>3️⃣ Model Evaluation</h3>
<ul>
  <li>Accuracy, Precision, Recall, F1-score</li>
  <li>ROC-AUC tracking across epochs</li>
</ul>

<h3>4️⃣ LLM-Based Report Generation</h3>
<ul>
  <li>Custom prompt templates</li>
  <li>Chain-of-thought driven pathology-style reporting</li>
  <li>Confidence scoring included in summary</li>
</ul>

<hr>

<h2>🔍 Sample Prediction Output</h2>

<pre>
Prediction: ALL
Probability: 97.13%

Generated Report:
The model detects morphological patterns consistent with Acute Lymphoblastic Leukemia.
High nuclear-to-cytoplasm ratios and lymphoblast features are prominent.
Recommended: CBC, bone marrow examination, and flow cytometry.
</pre>

<hr>

<h2>⚙️ Installation</h2>

<pre>
pip install -r requirements.txt
uv sync
</pre>

<hr>

<h2>▶️ Running Inference</h2>

<pre>
python main.py
</pre>

You will receive:
<ul>
  <li>Predicted class (ALL / HEM)</li>
  <li>Prediction probability</li>
  <li>LLM-generated diagnostic report</li>
</ul>

<hr>

<h2>📓 Notebooks</h2>

<p>
The <code>notebooks/main.ipynb</code> contains:
</p>

<ul>
  <li>Dataset inspection</li>
  <li>Training loop</li>
  <li>Metric visualization</li>
  <li>Prediction examples</li>
</ul>
<hr>

<h2>🛠️ Technical Implementation</h2>

<h3>🔧 Key Technologies</h3>

<ul>
  <li><b>Deep Learning Framework:</b> PyTorch</li>
  <li><b>Computer Vision:</b> torchvision, PIL</li>
  <li><b>Data Processing:</b> pandas, numpy</li>
  <li><b>Visualization:</b> matplotlib</li>
  <li><b>NLP:</b> LangChain (LLM-based report generation)</li>
  <li><b>Model Architecture:</b> DenseNet-121 (ImageNet-pretrained)</li>
</ul>

<h3>🏗 Design Patterns Used</h3>
<ul>
  <li><b>Modular Design:</b> Strict separation of concerns across data, model, inference, NLP</li>
  <li><b>Factory Pattern:</b> For clean model creation and switching architectures</li>
  <li><b>Pipeline Pattern:</b> Multi-step data processing and training pipeline</li>
  <li><b>Strategy Pattern:</b> Configurable training, augmentation, and optimization strategies</li>
</ul>

<h3>✅ Best Practices Implemented</h3>

<ul>
  <li>✔ Type hints throughout the codebase</li>
  <li>✔ Comprehensive logging system</li>
  <li>✔ Custom exception handling for debugging</li>
  <li>✔ Configuration & environment management</li>
  <li>✔ Model versioning for reproducibility</li>
  <li>✔ Seed setting for deterministic results</li>
</ul>

<hr>

<h2>📌 Future Improvements</h2>

<ul>
  <li>Multi-type leukemia classification</li>
  <li>Grad-CAM heatmap visualization</li>
  <li>Model compression for mobile inference</li>
  <li>More structured clinical reporting</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>

<p>Contributions are welcome! Follow the steps below:</p>

<ol>
  <li>Fork the repository</li>
  <li>Create a feature branch:<br><code>git checkout -b feature/AmazingFeature</code></li>
  <li>Commit your changes:<br><code>git commit -m "Add some AmazingFeature"</code></li>
  <li>Push to the branch:<br><code>git push origin feature/AmazingFeature</code></li>
  <li>Open a Pull Request</li>
</ol>

<hr>

<h2>📚 References</h2>

<ul>
  <li>
    Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). 
    <i>Densely Connected Convolutional Networks (DenseNet)</i>. 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
  </li>

  <li>
    C-NMC 2019 Dataset — 
    <a href="https://www.cancerimagingarchive.net/collection/c-nmc-2019/" target="_blank">
      The Cancer Imaging Archive (TCIA)
    </a>.<br>
    <i>Used as the primary dataset for ALL vs HEM blood smear classification.</i>
  </li>

  <li>
    Medical literature on Acute Lymphoblastic Leukemia diagnosis, cell morphology, 
    and hematopathology best practices.
  </li>

  <li>
    Best practices in medical imaging AI, model evaluation, and dataset handling.
  </li>
</ul>
