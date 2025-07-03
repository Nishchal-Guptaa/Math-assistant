# 1. Use official Python 3.12 image
FROM python:3.12-slim

# 2. Set work directory inside container
WORKDIR /app

# 3. Copy only necessary files
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy rest of the code (app.py, templates, etc.)
COPY . .

# 6. Expose Flask port (5000 default)
EXPOSE 5000

# 7. Run the app
CMD ["python", "app.py"]