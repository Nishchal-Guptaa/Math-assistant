from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import fitz  # PyMuPDF for PDF handling
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
import sympy as sp
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(path)
genai.configure(api_key=os.getenv("MATH_API"))


SYSTEM_INSTRUCTION = (
    "You are a helpful and reliable AI math assistant with advanced visualization capabilities."
    "You assist users in solving math problems, explaining concepts, and providing step-by-step solutions."
    "You can create various types of mathematical visualizations including 2D plots, 3D surfaces, statistical charts, and interactive graphs."
    "You should always ask relevant follow-up questions when the problem is vague or unclear."
    "If a user asks anything outside of the math domain (e.g., technology, travel, sports), respond with: 'I'm designed to assist only with math-related questions and cannot help with this topic.'"
    "Always be empathetic, respectful, and professional in tone, and never create panic or anxiety."
    "Do not speculate beyond what the data allows. If you are unsure, say so."
    "When explaining math concepts, use bullet points for clarity and provide visual aids when helpful."
    "Always end your response with a follow-up question like: 'Is there anything else you'd like me to help you with?😊'"
    "If a user greets you then respond with: 'Hello! I'm here to help you with your math-related questions and create visualizations. How can I assist you today? 😊'"
    "Do not generate any codes for answering any question. Only generate visualizations and explanations based on the provided mathematical expressions or problems."
)


def safe_eval(expression, x_vals):
    """Safely evaluate mathematical expressions"""
    # Replace common mathematical notation
    expression = expression.replace('^', '**')
    expression = expression.replace('ln', 'log')
    
    # Define safe mathematical functions
    safe_env = {
        "__builtins__": {},
        "x": x_vals,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "log": np.log,
        "ln": np.log,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "floor": np.floor,
        "ceil": np.ceil,
        "round": np.round,
        "np": np,
    }
    
    try:
        return eval(expression, safe_env)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def generate_advanced_plot(expression, plot_type="line", x_min=-10, x_max=10, color_theme="blue", **kwargs):
    """Generate advanced mathematical visualizations"""
    
    # Color schemes
    color_schemes = {
        "blue": ["#1f77b4", "#aec7e8", "#0d47a1"],
        "red": ["#d62728", "#ff9999", "#c62828"],
        "green": ["#2ca02c", "#98df8a", "#2e7d32"],
        "purple": ["#9467bd", "#c5b0d5", "#7b1fa2"],
        "rainbow": px.colors.qualitative.Vivid
    }
    
    colors = color_schemes.get(color_theme, color_schemes["blue"])
    
    try:
        if plot_type == "3d":
            return generate_3d_plot(expression, x_min, x_max, colors)
        elif plot_type == "parametric":
            return generate_parametric_plot(expression, x_min, x_max, colors)
        elif plot_type == "polar":
            return generate_polar_plot(expression, colors)
        elif plot_type == "contour":
            return generate_contour_plot(expression, x_min, x_max, colors)
        else:
            return generate_2d_plot(expression, plot_type, x_min, x_max, colors)
            
    except Exception as e:
        print(f"Plot generation error: {str(e)}")
        return None

def generate_2d_plot(expression, plot_type, x_min, x_max, colors):
    """Generate 2D plots with enhanced styling"""
    # Set style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0f0f23')
    
    # Generate data
    x = np.linspace(x_min, x_max, 1000)
    y = safe_eval(expression, x)
    
    if plot_type == "line":
        ax.plot(x, y, color=colors[0], linewidth=3, alpha=0.9, label=f'y = {expression}')
        ax.fill_between(x, y, alpha=0.2, color=colors[0])
    elif plot_type == "scatter":
        x_scatter = np.linspace(x_min, x_max, 100)
        y_scatter = safe_eval(expression, x_scatter)
        ax.scatter(x_scatter, y_scatter, c=colors[0], s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    elif plot_type == "bar":
        x_bar = np.linspace(x_min, x_max, 20)
        y_bar = safe_eval(expression, x_bar)
        ax.bar(x_bar, y_bar, color=colors[0], alpha=0.8, width=(x_max-x_min)/25)
    
    # Enhanced styling
    ax.grid(True, alpha=0.3, color='white', linestyle='--')
    ax.axhline(y=0, color='white', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='white', linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel('x', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('y', fontsize=12, color='white', fontweight='bold')
    ax.set_title(f'Graph of {expression}', fontsize=16, color='white', fontweight='bold', pad=20)
    
    ax.tick_params(colors='white', labelsize=10)
    ax.legend(fontsize=10, loc='best', framealpha=0.8)
    
    # Add subtle glow effect
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_3d_plot(expression, x_min, x_max, colors):
    """Generate 3D surface plots"""
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f23')
    
    # Create meshgrid for 3D surface
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(x_min, x_max, 50)
    X, Y = np.meshgrid(x, y)
    
    # Handle 3D expressions (assuming expression contains both x and y)
    try:
        # Replace x and y in expression for 3D evaluation
        expr_3d = expression.replace('x', 'X').replace('y', 'Y').replace('Y', 'Y')  # Keep Y as Y
        if 'Y' not in expr_3d and 'y' not in expression:
            # If no y in expression, create a surface using x and a function of x
            Z = safe_eval(expression.replace('x', 'X'), X) * np.ones_like(Y)
        else:
            # Evaluate 3D expression
            safe_env_3d = {
                "__builtins__": {},
                "X": X,
                "Y": Y,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "pi": np.pi,
                "np": np,
            }
            Z = eval(expr_3d, safe_env_3d)
    except:
        # Fallback: create a surface by rotating the 2D function
        Z = safe_eval(expression, np.sqrt(X**2 + Y**2))
    
    # Create surface plot with gradient colors
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                             linewidth=0, antialiased=True)
    
    # Add contour lines
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
    
    ax.set_xlabel('X', fontsize=12, color='white')
    ax.set_ylabel('Y', fontsize=12, color='white')
    ax.set_zlabel('Z', fontsize=12, color='white')
    ax.set_title(f'3D Surface: {expression}', fontsize=14, color='white', pad=20)
    
    # Color the axes
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_parametric_plot(expression, x_min, x_max, colors):
    """Generate parametric plots"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0f0f23')

    # Generate parameter t
    t = np.linspace(x_min, x_max, 1000)

    # Assuming the expression is in the form "f(t), g(t)"
    try:
        # Split the expression into x(t) and y(t)
        x_expr, y_expr = expression.split(',')
        x_vals = safe_eval(x_expr.strip(), t)
        y_vals = safe_eval(y_expr.strip(), t)

        # Plot the parametric curve
        ax.plot(x_vals, y_vals, color=colors[0], linewidth=3, alpha=0.9)
        ax.fill_between(x_vals, y_vals, alpha=0.2, color=colors[0])
        
    except Exception as e:
        raise ValueError(f"Error generating parametric plot: {str(e)}")

    # Enhanced styling
    ax.grid(True, alpha=0.3, color='white', linestyle='--')
    ax.axhline(y=0, color='white', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='white', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('x(t)', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('y(t)', fontsize=12, color='white', fontweight='bold')
    ax.set_title(f'Parametric Plot of {expression}', fontsize=16, color='white', fontweight='bold', pad=20)

    ax.tick_params(colors='white', labelsize=10)

    plt.tight_layout()
    return fig_to_base64(fig)

def generate_polar_plot(expression, colors):
    """Generate polar plots"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('#0f0f23')
    
    # Generate theta values
    theta = np.linspace(0, 2*np.pi, 1000)
    
    try:
        # Evaluate the expression for polar coordinates
        r = safe_eval(expression, theta)
        
        # Plot polar curve
        ax.plot(theta, r, color=colors[0], linewidth=3, alpha=0.9)
        
        # Fill the area under the curve
        ax.fill(theta, r, color=colors[0], alpha=0.3)
        
    except Exception as e:
        raise ValueError(f"Error generating polar plot: {str(e)}")
    
    # Styling
    ax.grid(True, alpha=0.3, color='white', linestyle='--')
    
    plt.title(f'Polar Plot of r = {expression}', 
              fontsize=16, color='white', fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_contour_plot(expression, x_min, x_max, colors):
    """Generate contour plots"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0f0f23')
    
    # Create grid
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(x_min, x_max, 100)
    X, Y = np.meshgrid(x, y)
    
    try:
        # Handle expressions for contour plots (expecting expression with x and y)
        Z = safe_eval(expression.replace('x', 'X').replace('y', 'Y'), X)
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, ax=ax, label='Function Value')
        
        # Add contour lines
        ax.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
        
    except Exception as e:
        raise ValueError(f"Error generating contour plot: {str(e)}")
    
    # Styling
    ax.grid(True, alpha=0.3, color='white', linestyle='--')
    ax.set_xlabel('x', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('y', fontsize=12, color='white', fontweight='bold')
    plt.title(f'Contour Plot of z = {expression}', 
              fontsize=16, color='white', fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_statistical_visualization(data_type, **kwargs):
    """Generate statistical visualizations"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0f0f23')
    
    if data_type == "normal_distribution":
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        ax.plot(x, y, color='#00d4ff', linewidth=3, label='Normal Distribution')
        ax.fill_between(x, y, alpha=0.3, color='#00d4ff')
        
    elif data_type == "histogram":
        # Generate sample data
        data = np.random.normal(0, 1, 1000)
        ax.hist(data, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='white')
        
    elif data_type == "box_plot":
        # Generate sample data
        data = [np.random.normal(0, std, 100) for std in range(1, 4)]
        ax.boxplot(data, patch_artist=True, 
                  boxprops=dict(facecolor='#51cf66', alpha=0.7),
                  medianprops=dict(color='white', linewidth=2))
    
    ax.grid(True, alpha=0.3, color='white', linestyle='--')
    ax.set_title(f'Statistical Visualization: {data_type.replace("_", " ").title()}', 
                fontsize=16, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#1a1a2e', dpi=150, bbox_inches='tight')
    plt.close(fig)  # Important: close figure to free memory
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# Initialize Gemini chat
chat = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction=SYSTEM_INSTRUCTION
).start_chat()

def extract_text_from_pdf(file_stream):
    """Extract text from PDF files"""
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question', '')
    
    response = f"{chat.send_message(question).text}"

    return jsonify({'response': response})


@app.route("/visualize", methods=["POST"])
def visualize():
    """Handle visualization requests"""
    data = request.get_json()
    
    expression = data.get("expression", "").strip()
    plot_type = data.get("graphType", "line")
    x_min = float(data.get("xMin", -10))
    x_max = float(data.get("xMax", 10))
    color_theme = data.get("color", "blue")

    if not expression:
        return jsonify({"error": "Please enter a valid mathematical expression."}), 400

    try:
        # Generate visualization
        graph_url = generate_advanced_plot(
            expression, plot_type, x_min, x_max, color_theme
        )
        
        if graph_url:
            # Generate explanation using Gemini
            explanation_prompt = f"Explain the mathematical function {expression} and describe its key properties, domain, range, and behavior."
            explanation_response = chat.send_message(explanation_prompt)
            
            return jsonify({
                "response": explanation_response.text,
                "graph": graph_url,
                "expression": expression,
                "type": plot_type
            })
        else:
            return jsonify({"error": "Unable to generate visualization. Please check your expression."}), 400
            
    except Exception as e:
        return jsonify({"error": f"Visualization error: {str(e)}"}), 500

@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    """Handle PDF analysis requests"""
    file = request.files.get("pdf")

    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    try:
        text = extract_text_from_pdf(file)
        if not text:
            return jsonify({"error": "The PDF appears to be empty or unreadable."}), 400

        response = chat.send_message(f"Please solve the following math problems and provide detailed step-by-step solutions:\n\n{text}")
        
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": f"PDF processing error: {str(e)}"}), 500

@app.route("/statistics", methods=["POST"])
def statistics():
    """Generate statistical visualizations"""
    data = request.get_json()
    stat_type = data.get("type", "normal_distribution")
    
    try:
        graph_url = generate_statistical_visualization(stat_type)
        
        if graph_url:
            explanation_prompt = f"Explain the {stat_type.replace('_', ' ')} and its mathematical properties."
            explanation_response = chat.send_message(explanation_prompt)
            
            return jsonify({
                "response": explanation_response.text,
                "graph": graph_url
            })
        else:
            return jsonify({"error": "Unable to generate statistical visualization."}), 400
            
    except Exception as e:
        return jsonify({"error": f"Statistics error: {str(e)}"}), 500

@app.route("/solve-equation", methods=["POST"])
def solve_equation():
    """Solve mathematical equations symbolically and return in standard math notation"""
    data = request.get_json()
    equation = data.get("equation", "").strip()

    if not equation:
        return jsonify({"error": "Please enter a valid equation."}), 400

    try:
        # Use SymPy for symbolic solving
        x = sp.Symbol('x')
        eq = sp.sympify(equation.replace("^", "**"))  # internal use
        solutions = sp.solve(eq, x)

        # Format readable equation
        pretty_eq = sp.pretty(eq, use_unicode=True)
        latex_eq = sp.latex(eq)

        # Format solutions nicely
        formatted_solutions = [sp.pretty(sol, use_unicode=True) for sol in solutions]
        latex_solutions = [sp.latex(sol) for sol in solutions]

        # Explanation using AI
        explanation_prompt = f"Explain step by step how to solve this equation: {equation.replace('**', '^')}, and verify the solutions: {solutions}"
        explanation_response = chat.send_message(explanation_prompt)

        return jsonify({
            "response": explanation_response.text,
            "equation_pretty": pretty_eq,
            "equation_latex": latex_eq,
            "solutions": formatted_solutions,
            "solutions_latex": latex_solutions
        })

    except Exception as e:
        return jsonify({"error": f"Equation solving error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)