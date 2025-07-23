# Step 1: Use an official Python runtime as a parent image
# Using a "slim" version keeps the final image size smaller
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the dependencies file first to leverage Docker's layer caching
# This means Docker won't reinstall dependencies unless requirements.txt changes
COPY requirements.txt .

# Step 4: Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application's code into the container
COPY . .

# Step 6: Expose the port the app runs on
# This tells Docker that the container listens on port 8000
EXPOSE 8000

# Step 7: Define the command to run your application
# We use uvicorn to run the FastAPI app.
# The host 0.0.0.0 is crucial to allow traffic from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]