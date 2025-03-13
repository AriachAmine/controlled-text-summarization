---
title: Controlled Text Summarization
emoji: 📈
colorFrom: yellow
colorTo: purple
sdk: gradio # VERY IMPORTANT: Specifies the framework (gradio, streamlit, docker, static)
sdk_version: 5.20.1 # Optional but recommended: Specify the SDK version
app_file: main.py # VERY IMPORTANT: The main Python file to run
pinned: false # Optional: Whether to pin the Space in your profile
license: mit # Optional: The license of your project (e.g., mit, apache-2.0, agpl-3.0)
---

# Creative Text Summarization with Style Control

A machine learning system that summarizes text in different stylistic variations (formal, informal, humorous, poetic) while preserving the content.

## Overview

This project creates an AI-powered text summarization system that not only condenses text but adapts its output to different stylistic preferences. It uses transformer models fine-tuned with style-specific summaries to generate summaries that match requested styles while maintaining accuracy.

## Features

- **Multiple Summary Styles**: Generate summaries in formal, informal, humorous, or poetic styles
- **Pre-trained Models**: Based on BART and other transformer architectures
- **User-friendly Interface**: Simple Gradio UI for interactive summary generation
- **Evaluation Metrics**: ROUGE and BLEU scores to evaluate summary quality

## Installation

1. Clone this repository:

```bash
git clone https://github.com/AriachAmine/controlled-text-summarization.git
cd controlled-text-summarization
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the Gemini API key:

```bash
# Linux/MacOS
export GEMINI_API_KEY="your-api-key-here"

# Windows
set GEMINI_API_KEY="your-api-key-here"
```

## Usage

### Running the Application

```bash
python main.py
```

This will:

1. Load the base model or a fine-tuned model (if available)
2. Prepare a dataset with stylized summaries (if needed)
3. Fine-tune the model on the prepared dataset (if no fine-tuned model exists)
4. Launch the Gradio interface for interactive summarization

### Using the Gradio Interface

1. Enter the text you want to summarize in the text box
2. Select your desired summary style from the dropdown (formal, informal, humorous, poetic)
3. Click "Submit" to generate the stylized summary

## How It Works

1. **Base Model**: Starts with a pre-trained text summarization model (BART)
2. **Style Training**: Fine-tunes the model on summaries with specific styles
3. **Style Control**: Uses style tokens to control output style during generation
4. **Evaluation**: Measures quality using ROUGE and BLEU metrics

## Project Structure

```
controlled-text-summarization/
├── main.py               # Main script to run the application
├── model.py              # Model loading and summarization functions
├── data.py               # Data preparation and processing
├── evaluation.py         # Metrics for evaluating summaries
├── ui.py                 # Gradio interface
├── requirements.txt      # Project dependencies
└── summarization_model/  # Directory for fine-tuned models (created after training)
```

## Dependencies

- torch, transformers: For model loading and fine-tuning
- gradio: For the user interface
- google-generativeai: For generating style-specific training data
- datasets, rouge_score, nltk: For data handling and evaluation

## Example

Input:

```
Scientists have discovered a new species of deep-sea fish that can withstand extreme pressure. The fish, found at depths of over 8,000 meters, has unique adaptations including specialized cell membranes and pressure-resistant proteins. This discovery may lead to new applications in biotechnology and materials science.
```

Output (Formal Style):

```
Researchers have identified a novel deep-sea fish species capable of surviving extreme pressures at depths exceeding 8,000 meters. The species exhibits specialized adaptations in cell membrane structure and pressure-resistant proteins, potentially offering valuable insights for biotechnology and materials science applications.
```

Output (Humorous Style):

```
Talk about a fish out of water... or rather, a fish VERY deep IN water! Scientists just found a super fish that laughs in the face of crushing ocean pressure. This deep-sea champion, chilling at 8,000 meters down, has fancy cell membranes and proteins that basically say "pressure, what pressure?" Scientists are already dreaming up ways to copy these deep-sea survival tricks for cool new tech!
```

## Future Improvements

- Add more styles (technical, narrative, etc.)
- Implement user feedback collection to improve models
- Add style strength control (slightly humorous vs. very humorous)
- Create a web API for integration with other applications

## License

[MIT License](LICENSE)
