# syntax=docker/dockerfile:1
FROM vggfiit/vgg-torch:2.0.0

# Update the APT-GET
RUN apt-get update --fix-missing
RUN apt-get install apt-transport-https
RUN apt-get install -y squashfuse
RUN apt-get install ca-certificates
RUN update-ca-certificates

# Install required packages
WORKDIR /setup
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Working dir
ENV PYTHONPATH=/workspace/stuFieldReg
WORKDIR /workspace/stuFieldReg