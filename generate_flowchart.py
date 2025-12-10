"""
Generate a visual flowchart of the RAG Model V2 process
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as mpatches

def create_rag_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # Define colors
    start_color = '#90EE90'  # Light green
    end_color = '#FFB6C1'    # Light pink
    data_color = '#E6F3FF'   # Light blue
    model_color = '#FFF4E6'  # Light orange
    index_color = '#E6FFE6'  # Light green
    query_color = '#E6E6FF'  # Light purple
    llm_color = '#FFE6FF'    # Light pink
    
    # Helper function to create rounded boxes
    def create_box(x, y, width, height, text, color, text_size=8):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=text_size, 
                weight='bold', wrap=True)
        return box
    
    # Helper function to create arrows
    def create_arrow(x1, y1, x2, y2, color='black', style='->'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=style, color=color, 
                               linewidth=2, mutation_scale=20)
        ax.add_patch(arrow)
    
    # Phase 1: Data Preparation
    y_start = 23
    create_box(5, y_start, 3, 0.6, 'START', start_color, 10)
    
    y = y_start - 1
    create_box(5, y, 3, 0.6, 'Load Amazon Product Data\n(10,002 products)', data_color)
    
    y -= 0.8
    create_box(5, y, 3, 0.6, 'Filter Products\nCheck Categories', data_color)
    
    y -= 0.8
    create_box(5, y, 3, 0.6, 'Download Product Images\nSave as {Uniq Id}.jpg', data_color)
    
    y -= 0.8
    create_box(5, y, 3, 0.6, 'Filter: Remove Missing Images\nFinal: 9,970 products', data_color)
    
    # Phase 2: CLIP Setup
    y -= 0.9
    create_box(5, y, 3, 0.6, 'Setup CLIP Model\nclip-vit-base-patch32', model_color)
    
    # Branch for encoding
    y -= 0.9
    create_box(2.5, y, 2.2, 0.6, 'Encode Product Text\nBatch Processing\n9,970 x 512 embeddings', model_color, 7)
    create_box(7.5, y, 2.2, 0.6, 'Encode Product Images\nBatch Processing\n9,970 x 512 embeddings', model_color, 7)
    
    y -= 0.9
    create_box(5, y, 3, 0.6, 'Combine Embeddings\nAverage & Re-normalize\n9,970 x 512', model_color)
    
    # Phase 3: Index Building
    y -= 0.9
    create_box(5, y, 3, 0.6, 'Build FAISS Index\nIndexFlatIP\n9,970 items', index_color)
    
    y -= 0.9
    create_box(5, y, 3, 0.6, 'Evaluate Zero-shot\nRecall@1=0.34, R@5=0.58, R@10=0.68', index_color)
    
    # Fine-tuning decision
    y -= 0.9
    create_box(5, y, 3, 0.6, 'Optional: Fine-tune?', '#FFE6E6', 9)
    
    # Fine-tuning path (left)
    y_ft = y - 0.9
    create_box(2, y_ft, 2, 0.6, 'Create Dataset\n5,000 samples', '#FFE6E6', 7)
    y_ft -= 0.8
    create_box(2, y_ft, 2, 0.6, 'Freeze CLIP\nBackbone', '#FFE6E6', 7)
    y_ft -= 0.8
    create_box(2, y_ft, 2, 0.6, 'Create ClipHead\nProjection Layers', '#FFE6E6', 7)
    y_ft -= 0.8
    create_box(2, y_ft, 2, 0.6, 'Train Head\n3 epochs\nContrastive Loss', '#FFE6E6', 7)
    y_ft -= 0.8
    create_box(2, y_ft, 2, 0.6, 'Save Head\nclip_head.pth', '#FFE6E6', 7)
    y_ft -= 0.8
    create_box(2, y_ft, 2, 0.6, 'Re-encode All\nFine-tuned Embeddings', '#FFE6E6', 7)
    y_ft -= 0.8
    create_box(2, y_ft, 2, 0.6, 'Evaluate Fine-tuned\nR@1=0.33, R@5=0.61, R@10=0.71', '#FFE6E6', 7)
    
    # Query processing (right path, or after fine-tuning)
    y_query = y - 0.9
    create_box(8, y_query, 2, 0.6, 'Skip Fine-tuning', '#E6E6FF', 7)
    
    # Merge point
    y_merge = min(y_ft, y_query) - 0.9
    create_box(5, y_merge, 3, 0.6, 'Query Processing', query_color)
    
    # Query type decision
    y -= 0.9
    y_merge = y_merge - 0.9
    create_box(5, y_merge, 3, 0.6, 'Query Type?', query_color, 9)
    
    # Branch for query types
    y_query_type = y_merge - 0.9
    create_box(2.5, y_query_type, 2.2, 0.6, 'Text Query\nEncode with CLIP\nFine-tuned head', query_color, 7)
    create_box(7.5, y_query_type, 2.2, 0.6, 'Image Query\nEncode with CLIP\nFine-tuned head', query_color, 7)
    
    # Merge to retrieval
    y_retrieve = y_query_type - 0.9
    create_box(5, y_retrieve, 3, 0.6, 'Retrieve Top-K Products\nFAISS Search\nk=5 default', query_color)
    
    # RAG Pipeline
    y_rag = y_retrieve - 0.9
    create_box(5, y_rag, 3, 0.6, 'Format RAG Context\nExtract: Name, Brand,\nCategory, Price, Description', llm_color)
    
    y_rag -= 0.9
    create_box(5, y_rag, 3, 0.6, 'Build RAG Prompt\nInclude Instructions\nUser Query\nRetrieved Products', llm_color)
    
    y_rag -= 0.9
    create_box(5, y_rag, 3, 0.6, 'Call LLM\nGPT-4.1-mini\nGenerate Answer', llm_color)
    
    y_rag -= 0.9
    create_box(5, y_rag, 3, 0.6, 'Return Answer\n+ Retrieved Products', '#E6FFE6')
    
    y_end = y_rag - 0.9
    create_box(5, y_end, 3, 0.6, 'END', end_color, 10)
    
    # Draw arrows - Main flow
    y_pos = y_start
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 2.5, y_pos-0.7)
    create_arrow(5, y_pos-0.3, 7.5, y_pos-0.7)
    y_pos -= 1
    create_arrow(2.5, y_pos-0.3, 5, y_pos-0.7)
    create_arrow(7.5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    create_arrow(5, y_pos-0.3, 5, y_pos-0.7)
    y_pos -= 1
    
    # Fine-tuning arrows
    create_arrow(5, y_pos-0.3, 2, y_pos-0.7)
    create_arrow(5, y_pos-0.3, 8, y_pos-0.7)
    
    # Fine-tuning path arrows
    y_ft_arrow = y_pos - 0.9
    for _ in range(6):
        create_arrow(2, y_ft_arrow-0.3, 2, y_ft_arrow-0.7)
        y_ft_arrow -= 0.8
    
    # Skip path arrow
    create_arrow(8, y_query-0.3, 5, y_merge+0.3)
    
    # Fine-tuned path arrow
    create_arrow(2, y_ft+0.3, 5, y_merge+0.3)
    
    # Query processing arrows
    y_merge_arrow = y_merge
    create_arrow(5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(5, y_merge_arrow-0.3, 2.5, y_merge_arrow-0.7)
    create_arrow(5, y_merge_arrow-0.3, 7.5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(2.5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    create_arrow(7.5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    y_merge_arrow -= 0.9
    create_arrow(5, y_merge_arrow-0.3, 5, y_merge_arrow-0.7)
    
    # Title
    ax.text(5, 23.5, 'RAG Model V2 - Complete Process Flowchart', 
           ha='center', va='center', fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('RAG_Process_Flowchart.png', dpi=300, bbox_inches='tight')
    print("Flowchart saved as 'RAG_Process_Flowchart.png'")
    plt.show()

if __name__ == '__main__':
    create_rag_flowchart()

