FROM gitlab-registry.nrp-nautilus.io/cboettig/images
WORKDIR /app

COPY . .

# huggingface uses port 7860 by default
CMD streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --server.fileWatcherType none
