# 🌙 Dream Analyzer

> An NLP-based web application where users can describe their dreams and an ensemble model classifies them according to the emotion associated with the dream.

---

## 📖 Problem Statement / Motivation

Dreams can carry strong emotional signals, yet interpreting those signals is subjective and time-consuming. Dream Analyzer automates that process: users write a free-text description of their dream, and a trained ensemble of NLP models predicts the underlying emotion (e.g., *happy*, *sad*, *neutral*, *nightmare*). The app also maintains a personal dream journal, surfaces trend analytics over time, and plays ambient music that matches the detected mood — turning dream exploration into a rich, interactive experience.

---

## ✨ Key Features

- **Emotion classification** — ensemble of Naive Bayes, LightGBM, and a TextCNN model blended with a Logistic Regression meta-learner
- **Dream journal** — save, search, and filter entries by date, emotion, tags, or lucidity
- **Analysis dashboard** — interactive charts (Plotly): emotion timeline, distribution pie, confidence trends, lucid-dream monthly breakdown
- **Mood soundtrack** — auto-plays ambient music matched to the predicted emotion
- **Dynamic theming** — app colour scheme adapts based on the last detected emotion
- **t-SNE / KMeans cluster visualisation** — pre-computed cluster view of the training data

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Web framework | [Streamlit](https://streamlit.io/) |
| Deep learning | [PyTorch](https://pytorch.org/) (TextCNN) |
| Classical ML | Scikit-learn (Naive Bayes, Logistic Regression), [LightGBM](https://lightgbm.readthedocs.io/) |
| NLP / features | TF-IDF, custom vocabulary + tokeniser |
| Data | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn, WordCloud |
| Serialisation | Joblib, PyTorch `torch.save` |
| Language | Python 3.x |

---

## 📂 Repository Structure

```
Dream-Analyzer/
├── app.py                          # Streamlit application entry point
├── train.ipynb                     # Full training pipeline (ensemble)
├── nb_cnn.ipynb                    # TextCNN training notebook
├── eda.ipynb                       # Exploratory data analysis notebook
├── dreams_labeled_balanced.csv     # Labelled & balanced training dataset
├── final_predictions_with_tsne.csv # t-SNE + cluster results (CSV)
├── final_predictions_with_tsne.json# t-SNE + cluster results (JSON)
├── cnn_model.pt                    # Trained TextCNN weights (PyTorch)
├── nb_pipeline.pkl                 # Trained Naive Bayes pipeline
├── lgb_model.pkl                   # Trained LightGBM model
├── log_blender.pkl                 # Trained Logistic Regression blender
├── tfidf.pkl                       # Fitted TF-IDF vectoriser
├── label_encoder.pkl               # Fitted LabelEncoder
├── word2idx.pkl                    # Vocabulary → index mapping
├── emotion_music/                  # Ambient audio files per emotion
│   ├── happy.mp3
│   ├── sad.mp3
│   ├── neutral.mp3
│   └── nightmare.mp3
└── requirements.txt                # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shradha0511/Dream-Analyzer.git
cd Dream-Analyzer
```

### 2. Create a virtual environment

**Using `venv` (recommended):**

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

**Using `conda`:**

```bash
conda create -n dream-analyzer python=3.10
conda activate dream-analyzer
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` is empty or missing, install the core packages manually:
> ```bash
> pip install streamlit pandas numpy torch torchvision scikit-learn lightgbm plotly matplotlib seaborn joblib Pillow wordcloud
> ```

---

## 🚀 How to Run

### Run the Streamlit web app

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

#### Pages available in the sidebar

| Page | Description |
|---|---|
| **Journal Entry** | Enter a dream description → get an emotion prediction, confidence chart, and a matching soundtrack |
| **Dream Archive** | Browse, search, and filter all saved dreams |
| **Analysis Dashboard** | Visualise emotion trends, distribution, confidence, and lucidity over time |
| **Settings** | App-level configuration (TODO: expand as needed) |

#### Example usage

1. Navigate to **Journal Entry**.
2. Fill in the dream date, an optional title and tags.
3. In the **Dream Description** box, enter something like:
   > *"I was flying over a beautiful golden city, completely free and filled with joy."*
4. Click **Analyse & Save Dream**.
5. The app displays the predicted emotion (e.g., `happy`), a bar chart of class probabilities, and begins playing ambient music.

### Run the training / analysis notebooks

```bash
jupyter notebook
```

Open any of the following:

| Notebook | Purpose |
|---|---|
| `eda.ipynb` | Exploratory data analysis — label distribution, word counts, word clouds |
| `nb_cnn.ipynb` | TextCNN training (PyTorch) |
| `train.ipynb` | Full ensemble training pipeline (NB + LightGBM + CNN → blender) |

---

## 🤖 Model / NLP Pipeline Summary

### Dataset

- File: `dreams_labeled_balanced.csv`
- Columns: `content` (dream text), `label` (emotion category)
- The dataset is pre-balanced across emotion classes (verified via EDA notebook)
- Emotion classes include: `happy`, `sad`, `neutral`, `nightmare`, and potentially others

### Preprocessing

1. Remove parenthesised text (e.g., date annotations such as `(1960-05-04)`)
2. Strip non-alphabetic characters
3. Lowercase and strip whitespace
4. Tokenise by whitespace; truncate / pad sequences to a maximum length of 100 tokens

### Ensemble Approach

The system uses a **stacked ensemble (blending)**:

| Model | Feature input | Library |
|---|---|---|
| **Naive Bayes** | TF-IDF bag-of-words | Scikit-learn `MultinomialNB` + `Pipeline` |
| **LightGBM** | TF-IDF features | `LGBMClassifier` |
| **TextCNN** | Numericalized token sequences (embedding dim 128, filters of size 3/4/5, 100 filters each, global max-pool) | PyTorch |
| **Blender** | Concatenated softmax probability vectors from all three base models | Scikit-learn `LogisticRegression` |

At inference time:
1. The dream text is cleaned and transformed with the fitted TF-IDF vectoriser and word→index mapping.
2. Each base model produces class probability vectors.
3. The three probability vectors are horizontally stacked and fed to the Logistic Regression blender.
4. The blender outputs the final emotion label and class probabilities.

### Evaluation

The training notebooks include:

- `accuracy_score`, `f1_score`, `classification_report` (Scikit-learn)
- Confusion matrix visualisation (Seaborn heatmap)
- t-SNE 2-D projection of blended features with KMeans cluster colouring (results saved to `final_predictions_with_tsne.csv` / `.json`)

---

## 🔧 Configuration

| Item | Default | Notes |
|---|---|---|
| Model artefacts | Project root directory | All `.pkl` / `.pt` files must be in the same directory as `app.py` |
| Dream journal storage | `dream_journal.csv` (auto-created) | Local CSV; no external database required |
| Audio files | `emotion_music/<emotion>.mp3` | Add additional `.mp3` files here for other emotion classes |
| Max sequence length | 100 tokens | Hard-coded in `encode_text()` in `app.py` |
| CNN embedding dimension | 128 | Defined during model construction in `app.py` |

No environment variables or secrets are required to run the application.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add relevant tests or notebook examples.
4. Commit with a clear message: `git commit -m "Add: description of change"`
5. Push to your fork and open a Pull Request against `main`.

Please keep pull requests focused and describe the motivation clearly in the PR description.

---

## 📄 License

No license file is currently present in this repository. All rights are reserved by the author(s) unless otherwise stated. If you intend to use or build on this code, please contact the repository owner for permission.

---

## 🙏 Acknowledgements

- Dream text data sourced and labelled for emotion classification (see `dreams_labeled_balanced.csv`)
- [Streamlit](https://streamlit.io/) for the rapid web-app framework
- [PyTorch](https://pytorch.org/) for the deep learning infrastructure
- [LightGBM](https://lightgbm.readthedocs.io/) for the gradient-boosting classifier
- [Plotly](https://plotly.com/python/) for interactive visualisations
- The open-source Python data-science ecosystem (NumPy, Pandas, Scikit-learn)
