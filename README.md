# 🧠 Math Assistant

This project is an **AI-powered math assistant and visualization tool** built using **Flask** and **Google Gemini API**. It helps users:

* Solve mathematical expressions and equations.
* Generate advanced 2D/3D plots and statistical visualizations.
* Analyze PDF files with embedded math content.
* Receive AI-generated step-by-step explanations and insights.

---

## 🚀 Features

* 🧮 Solve symbolic equations using **SymPy**.
* 📊 Generate visualizations: line, scatter, bar, parametric, polar, 3D surface, and contour plots using **Matplotlib**, **Plotly**, and **Seaborn**.
* 📈 Statistical visualizations: normal distribution, histograms, box plots.
* 📎 Upload PDFs and extract math-related content using **PyMuPDF** (`fitz`).
* 💡 AI explanations with **Gemini (Google Generative AI)**.
* 🌐 Clean Flask-based API for front-end or external usage.

---

## 📁 Directory Structure

```
project/
│
├── templates/
│   └── index.html           # Frontend UI
├── .env                     # Environment file for API key
├── app.py                   # Main Flask app
├── requirements.txt         # Python dependencies
└── README.md                # You're reading this!
```

---

## 🧪 Sample API Endpoints

### 1. `/ask`

* **Method**: POST
* **Description**: Ask any math-related question to Gemini.
* **Body**: `question: string`
* **Returns**: AI-generated response.

---

### 2. `/visualize`

* **Method**: POST
* **Description**: Generate a visualization for a mathematical expression.
* **Body**:

```json
{
  "expression": "sin(x)",
  "graphType": "line",
  "xMin": -10,
  "xMax": 10,
  "color": "blue"
}
```

* **Returns**: Plot image in base64 + explanation.

---

### 3. `/ask-pdf`

* **Method**: POST
* **Description**: Upload a PDF file for mathematical problem extraction and solving.
* **Body**: Form-data with a `pdf` file.
* **Returns**: AI-generated analysis of the file.

---

### 4. `/statistics`

* **Method**: POST
* **Description**: Generate a statistical visualization.
* **Body**:

```json
{ "type": "normal_distribution" }
```

* **Options**: `normal_distribution`, `histogram`, `box_plot`

---

### 5. `/solve-equation`

* **Method**: POST
* **Description**: Solve symbolic equations.
* **Body**:

```json
{ "equation": "x^2 - 4 = 0" }
```

* **Returns**: Solutions, LaTeX format, and explanation.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Nishchal-Guptaa/Math-assistant
cd Math-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the root directory and add:

```
MATH_API=your_gemini_api_key_here
```

### 5. Run the app

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📦 Requirements

Dependencies include:

* Flask
* Python-dotenv
* PyMuPDF (`fitz`)
* NumPy, Matplotlib, Seaborn
* Plotly
* SymPy
* SciPy
* google-generativeai

---

## 🤖 AI Behavior Design

The assistant:

* Responds **only to math-related queries**.
* Creates **visual aids** for better understanding.
* Maintains a **polite, clear, and concise tone**.
* Gracefully handles vague or invalid inputs.
* Rejects non-math requests politely.

---

## 🔒 Security Notes

* Evaluation of expressions is sandboxed using a restricted `eval()` context.
* File handling is limited to PDFs.
* Set `debug=False` in production environments.

---

## 📌 Example Use Cases

* Education: Math tutoring and homework help.
* Research: Quick visualization and analysis.
* Engineering: Graphing functions and systems.
* Stats: Visual insight into data distributions.

---


## 🙋‍♀️ Need Help?

Is there anything else you'd like me to help you with? 😊
Feel free to raise an issue or start a discussion!

### View Documentation

View the documentation provided [https://emerald-lint-42b.notion.site/Math-AI-Assistant-Documentation-22588697fbea8055939ef8c9243b6ed6?pvs=141](Here).
