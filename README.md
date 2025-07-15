<div align="center">
  <img src="https://i.ytimg.com/vi/ClF55GE7zPI/maxresdefault.jpg" alt="Squeezie's Video Thumbnail" width="600"/>
  <h1>YouTube Comment Sentiment Analysis - MLOps Project</h1>
  <p>
    An end-to-end MLOps project that fetches YouTube comments, analyzes their sentiment, trains a model, and serves it via a live Streamlit application.
  </p>
  
  <p>
    <a href="https://youtube-sentiment-mlops.streamlit.app/">
      <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
    </a>
    &nbsp;
    <a href="https://github.com/BryanBradfo/youtube-sentiment-mlops/actions/workflows/main.yml">
      <img src="https://github.com/BryanBradfo/youtube-sentiment-mlops/actions/workflows/main.yml/badge.svg" alt="CI Pipeline Status">
    </a>
    &nbsp;
    <a href="https://github.com/BryanBradfo/youtube-sentiment-mlops/releases/latest">
      <img src="https://img.shields.io/github/v/release/BryanBradfo/youtube-sentiment-mlops" alt="Latest Release">
    </a>
  </p>
</div>

---

## 🚀 The Project

This repository contains a complete, end-to-end Machine Learning Operations (MLOps) project. The goal is to analyze the sentiment of comments from a YouTube video—in this case, Squeezie's famous "QUI EST L'IMPOSTEUR ?"—and provide an interactive web application for on-demand analysis of any YouTube video.

This project isn't just about building a model; it's about building a **robust, automated, and reproducible system** around it.

### ✨ Key Features

- **Automated Data Pipeline**: Fetches, preprocesses, and annotates data automatically using `DVC`.
- **CI/CD with GitHub Actions**:
  - **Continuous Integration (CI)**: Every push to the `main` branch automatically lints, tests, and validates the entire data pipeline.
  - **Continuous Delivery (CD)**: Pushing a new Git tag (e.g., `v1.1`) automatically trains the model, packages it, and creates a new release on GitHub.
- **Interactive Web App**: A [**Streamlit application**](https://youtube-sentiment-mlops.streamlit.app/) that allows anyone to analyze a YouTube video's comments using the latest trained model.
- **Version Control for Data & Models**: `DVC` and `MLflow` ensure that every component—code, data, and models—is versioned and tracked.

## 🛠️ Tech Stack

This project leverages a modern MLOps stack:

| Component             | Tool                                                                                             | Purpose                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Data & Pipeline**   | <img src="https://dvc.org/social-share.png" width="20"/> **DVC**                                      | Data Version Control & Pipeline Orchestration.            |
| **Experiment Tracking**| <img src="https://miro.medium.com/v2/resize:fit:528/0*4Kw51eGc74EsFLSs.png" width="20"/> **MLflow** | Tracking experiments, logging models, and the Model Registry. |
| **CI/CD**             | <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"/> **GitHub Actions** | Automated testing and release creation.                     |
| **Web Application**   | <img src="https://images.seeklogo.com/logo-png/44/2/streamlit-logo-png_seeklogo-441815.png" width="20"/> **Streamlit** | Building and deploying the interactive app.                 |
| **Core Libraries**    | `scikit-learn`, `pandas`, `PyTorch`, `transformers`                                                | Model training, data manipulation, and NLP tasks.           |

## 🏛️ Project Architecture

The project is designed with a clear separation of concerns, following best MLOps practices.

  <!-- TODO: Create and upload an architecture diagram -->

1.  **Development & CI Loop (Left)**:
    - Code is written and pushed to GitHub.
    - The CI pipeline (`main.yml`) runs tests and linters to ensure code quality.
2.  **Training & CD Loop (Middle)**:
    - A Git tag (e.g., `v1.0`) triggers the CD pipeline (`release.yml`).
    - The full `DVC` pipeline runs, from data fetching to model training.
    - The final model and its metrics are packaged and published as a **GitHub Release**.
3.  **Inference Loop (Right)**:
    - The **Streamlit app** automatically fetches the **latest release** from GitHub.
    - It loads the pre-trained model.
    - The user provides a YouTube URL, and the app performs fast inference to display sentiment analysis results.

## ⚙️ How It Works: The DVC Pipeline

The core logic is orchestrated by DVC. You can see the stages defined in `dvc.yaml`. Running `dvc repro` executes the following steps:

1.  `fetch`: Fetches thousands of comments from a specific YouTube video using the YouTube Data API.
2.  `preprocess`: Cleans the text data by removing URLs, mentions, and normalizing emojis.
3.  `annotate`: Uses a pre-trained Hugging Face model (`twitter-xlm-roberta-base-sentiment`) to assign an initial sentiment label (Positive, Negative, Neutral) to each comment. This serves as our ground truth.
4.  `train`: Trains a `scikit-learn` pipeline, which includes a `TfidfVectorizer` and a `StackingClassifier`, on the annotated data. The experiment is logged with **MLflow**.

## 🚀 Getting Started Locally

Want to run the project on your own machine? Here's how.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
- A **YouTube Data API v3 key** from the [Google Cloud Console](https://console.cloud.google.com/).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BryanBradfo/youtube-sentiment-mlops.git
    cd youtube-sentiment-mlops
    ```

2.  **Set up the Conda environment:**
    This project uses a Conda environment defined in `ci_environment.yml`.
    ```bash
    conda env create -f ci_environment.yml
    conda activate sentiment-mlops
    ```

3.  **Configure your API Key:**
    Create a file named `.env` in the root directory and add your API key:
    ```
    # .env
    YOUTUBE_API_KEY="YOUR_API_KEY_HERE"
    ```

### Running the Pipeline

To run the entire pipeline from start to finish, simply use the DVC command:

```bash
dvc repro
```

This will generate all the data files and log a new MLflow experiment in the `mlruns/` directory.

### Viewing Experiments

To see the results of your training runs, launch the MLflow UI:

```bash
mlflow ui
```

Then, open your browser to `http://127.0.0.1:5000`.

## 🤝 Contributing

Contributions are welcome! If you have an idea for an improvement or find a bug, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

---
<div align="center">
  Made with ❤️ and MLOps principles.
</div>