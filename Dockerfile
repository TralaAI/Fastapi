FROM python:3.13-slim

WORKDIR /app

# Install prerequisites
RUN apt-get update && apt-get install -y curl gnupg apt-transport-https unixodbc unixodbc-dev

# Download and trust the Microsoft GPG key
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg

# Add the Microsoft repository for Debian 12
RUN echo "deb [arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/msprod.list

# Install msodbcsql17
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]