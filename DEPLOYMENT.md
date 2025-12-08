# Streamlit Cloud Deployment Guide

This guide will help you deploy the Ask Lumi application to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at https://streamlit.io/cloud)
4. An OpenAI API key

## Step 1: Prepare Your Repository

Ensure your repository contains all necessary files:
- `app.py` - Main Streamlit application
- `models.py` - Model loading module
- `retrieval.py` - Retrieval module
- `rag_pipeline.py` - RAG pipeline module
- `requirements.txt` - Python dependencies
- `amazon_processed.csv` - Product dataset
- `clip_head.pth` - Fine-tuned CLIP head weights
- `images/` - Product images directory (~347MB)
- `.streamlit/config.toml` - Streamlit configuration
- `Lumi_logo.png` - Logo image
- `Lumi.mov` - Splash screen video (optional)

**Note:** The total repository size should be under 1GB for the free tier.

## Step 2: Push to GitHub

If you haven't already, push your code to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch (usually `main`)
5. Set the main file path to: `app.py`
6. Click "Deploy"

## Step 4: Configure Secrets

After deployment, you need to add your OpenAI API key:

1. In your Streamlit Cloud dashboard, click on your app
2. Click "Settings" (⚙️ icon)
3. Go to "Secrets" tab
4. Add the following:

```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```

5. Click "Save"
6. The app will automatically redeploy

## Step 5: First Deployment

The first deployment may take 10-15 minutes because:
- Dependencies need to be installed
- Models need to be downloaded
- Embeddings need to be computed (if cache doesn't exist)

**Important:** The first run will compute embeddings which takes 5-10 minutes. Subsequent runs will be faster due to caching.

## Troubleshooting

### App fails to start

1. **Check logs:** Click "Manage app" → "Logs" to see error messages
2. **Verify requirements.txt:** Ensure all dependencies are listed
3. **Check file paths:** Ensure all required files are in the repository
4. **Verify secrets:** Make sure `OPENAI_API_KEY` is set correctly

### Out of memory errors

- Streamlit Cloud free tier has limited memory
- If you encounter memory issues, consider:
  - Using smaller models
  - Reducing the number of products
  - Optimizing the embedding cache

### Large repository size

- The `images/` folder is ~347MB
- Total repository should be under 1GB
- If you exceed limits, consider:
  - Using Git LFS for large files
  - Hosting images externally (S3, Cloudinary, etc.)
  - Compressing images

### Slow loading times

- First deployment is slow (downloading models, computing embeddings)
- Subsequent runs use cached embeddings
- Consider pre-computing and committing `embeddings_cache.pkl` to speed up deployment

## Local Testing Before Deployment

Test your app locally first:

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (optional - app will prompt if not set)
export OPENAI_API_KEY=your-key-here

# Run the app
streamlit run app.py
```

## Environment Variables

The app checks for API keys in this order:
1. Streamlit secrets (`st.secrets['OPENAI_API_KEY']`) - **for cloud deployment**
2. Environment variable (`OPENAI_API_KEY`) - for local development
3. Session state - fallback for manual entry

## File Structure

```
.
├── app.py                 # Main Streamlit app
├── models.py              # Model loading
├── retrieval.py           # Product retrieval
├── rag_pipeline.py        # RAG pipeline
├── requirements.txt       # Dependencies
├── .streamlit/
│   └── config.toml        # Streamlit config
├── amazon_processed.csv   # Product data
├── clip_head.pth         # Model weights
├── images/               # Product images (~347MB)
├── Lumi_logo.png        # Logo
└── Lumi.mov             # Splash video
```

## Additional Notes

- The app uses `@st.cache_resource` to cache models and embeddings
- First-time users will see a splash screen
- API keys are securely stored in Streamlit Cloud secrets
- The app is responsive and works on mobile devices

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Test locally first
3. Verify all files are committed to GitHub
4. Ensure secrets are configured correctly

