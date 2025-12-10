# RAG Model V2 - Process Flowchart

```mermaid
flowchart TD
    Start([Start]) --> LoadData[Load Amazon Product Data<br/>amazon_processed.csv<br/>10,002 products]
    
    LoadData --> FilterData[Filter Products<br/>Check Main Categories]
    
    FilterData --> DownloadImages[Download Product Images<br/>Create images/ directory<br/>Save as {Uniq Id}.jpg]
    
    DownloadImages --> FilterImages[Filter Products<br/>Remove rows with missing images<br/>Final: 9,970 products]
    
    FilterImages --> SetupCLIP[Setup CLIP Model<br/>openai/clip-vit-base-patch32<br/>Load processor & model<br/>Set device: CUDA/CPU]
    
    SetupCLIP --> EncodeText[Encode Product Text<br/>Batch processing<br/>Generate text embeddings<br/>Shape: 9,970 x 512<br/>L2 normalized]
    
    SetupCLIP --> EncodeImages[Encode Product Images<br/>Batch processing<br/>Generate image embeddings<br/>Shape: 9,970 x 512<br/>L2 normalized]
    
    EncodeText --> CombineEmbs[Combine Embeddings<br/>Average: text_embs + image_embs / 2<br/>Re-normalize<br/>Shape: 9,970 x 512]
    
    EncodeImages --> CombineEmbs
    
    CombineEmbs --> BuildFAISS[Build FAISS Index<br/>IndexFlatIP<br/>Add combined embeddings<br/>Index size: 9,970]
    
    BuildFAISS --> EvalZeroShot{Evaluate<br/>Zero-shot Performance}
    
    EvalZeroShot --> ComputeRecall[Compute Recall Metrics<br/>Recall@1, @5, @10<br/>Text-to-Image retrieval<br/>Results: R@1=0.34, R@5=0.58, R@10=0.68]
    
    ComputeRecall --> FineTuneDecision{Optional:<br/>Fine-tune?}
    
    FineTuneDecision -->|Yes| CreateDataset[Create Training Dataset<br/>AmazonClipDataset<br/>Limit to 5,000 samples]
    
    FineTuneDecision -->|No| QueryProcessing[Query Processing]
    
    CreateDataset --> FreezeCLIP[Freeze CLIP Backbone<br/>Set requires_grad=False]
    
    FreezeCLIP --> CreateHead[Create ClipHead<br/>Linear projection layers<br/>Image & Text projectors<br/>Logit scale parameter]
    
    CreateHead --> TrainHead[Train Head<br/>Contrastive Loss<br/>3 epochs<br/>AdamW optimizer<br/>LR: 1e-3]
    
    TrainHead --> SaveHead[Save Fine-tuned Head<br/>clip_head.pth]
    
    SaveHead --> ComputeFTEmbs[Compute Fine-tuned Embeddings<br/>Re-encode all products<br/>Using trained head]
    
    ComputeFTEmbs --> EvalFineTuned[Evaluate Fine-tuned<br/>Recall@1=0.33, R@5=0.61, R@10=0.71]
    
    EvalFineTuned --> QueryProcessing
    
    QueryProcessing --> QueryType{Query Type?}
    
    QueryType -->|Text Query| EncodeQueryText[Encode Text Query<br/>CLIP text encoder<br/>Fine-tuned head projection<br/>L2 normalize]
    
    QueryType -->|Image Query| EncodeQueryImage[Encode Image Query<br/>CLIP image encoder<br/>Fine-tuned head projection<br/>L2 normalize]
    
    EncodeQueryText --> Retrieve[Retrieve Top-K Products<br/>Cosine similarity search<br/>FAISS index search<br/>k=5 default]
    
    EncodeQueryImage --> Retrieve
    
    Retrieve --> FormatContext[Format RAG Context<br/>Extract product info:<br/>- Name, Brand, Category<br/>- Price, Description<br/>- Truncate to 600 chars]
    
    FormatContext --> BuildPrompt[Build RAG Prompt<br/>Include instructions<br/>User query<br/>Retrieved products context]
    
    BuildPrompt --> CallLLM[Call LLM<br/>OpenAI GPT-4.1-mini<br/>Generate answer]
    
    CallLLM --> ReturnAnswer[Return Answer<br/>+ Retrieved Products]
    
    ReturnAnswer --> End([End])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style LoadData fill:#E6F3FF
    style SetupCLIP fill:#FFF4E6
    style BuildFAISS fill:#E6FFE6
    style FineTuneDecision fill:#FFE6E6
    style QueryType fill:#E6E6FF
    style CallLLM fill:#FFE6FF
    style ReturnAnswer fill:#E6FFE6
```

## Process Overview

### Phase 1: Data Preparation
1. **Load Data**: Load Amazon product dataset (10,002 products)
2. **Download Images**: Download product images from URLs
3. **Filter**: Remove products without valid images (final: 9,970 products)

### Phase 2: Embedding Generation
1. **Setup CLIP**: Load pre-trained CLIP model (ViT-Base-Patch32)
2. **Encode Text**: Generate text embeddings for product descriptions
3. **Encode Images**: Generate image embeddings for product images
4. **Combine**: Average and normalize text + image embeddings

### Phase 3: Index Building
1. **Build FAISS Index**: Create similarity search index using combined embeddings
2. **Evaluate**: Compute zero-shot retrieval performance metrics

### Phase 4: Fine-tuning (Optional)
1. **Create Dataset**: Prepare training dataset from product pairs
2. **Freeze Backbone**: Keep CLIP base model frozen
3. **Train Head**: Train projection head with contrastive loss
4. **Re-encode**: Generate fine-tuned embeddings for all products
5. **Evaluate**: Compare fine-tuned vs zero-shot performance

### Phase 5: Query Processing & RAG
1. **Query Encoding**: Encode user query (text or image) using CLIP
2. **Retrieval**: Search FAISS index for top-k similar products
3. **Context Formatting**: Format retrieved products into structured context
4. **Prompt Building**: Create RAG prompt with instructions and context
5. **LLM Generation**: Generate answer using GPT-4.1-mini
6. **Return Results**: Return answer + retrieved products

## Key Components

- **CLIP Model**: Multimodal encoder for text and images
- **FAISS Index**: Efficient similarity search
- **Fine-tuned Head**: Projection layers for domain adaptation
- **RAG Pipeline**: Retrieval-Augmented Generation for product recommendations
- **LLM**: GPT-4.1-mini for natural language responses

