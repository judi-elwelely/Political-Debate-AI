# Political Debate AI

A Gradio application that simulates a political debate between two AI agents:

- **Government Agent:** presents arguments supporting the government's policies.
- **Opposition Agent:** critiques those policies and presents alternative viewpoints.

The user specifies the country and the debate length. The application uses Groq-hosted language models and limits the debate to a maximum of 10 rounds.

## Features

- Two AI agents with distinct roles: Government and Opposition.
- Country specified by the user.
- Configurable debate length, with a maximum of 10 rounds.
- Gradio web interface.
- API key loaded from an environment file rather than hard-coded in the source code.

## Requirements

- Python 3.10 or later
- A Groq API key

## Installation

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Create a file named `.env` in the project folder and add your Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Replace `your_groq_api_key_here` with your own key. **Never upload the `.env` file or share your API key.**

## Run the application

Start the application with:

```bash
python appDebate.py
```

Then open the local Gradio URL printed in the terminal (usually `http://127.0.0.1:7860`).

## Important notes

- The debate is limited to 10 rounds.
- AI-generated claims and statistics may be inaccurate or outdated. Verify important facts using reliable, up-to-date sources.
- The application requires a valid Groq API key to call the models.

## Security

The `.env` file is excluded from Git using `.gitignore`. Keep API keys private and do not commit them to the repository.
