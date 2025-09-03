import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
primary_color = '#2E86AB'
secondary_color = '#A23B72'
accent_color = '#F18F01'
text_color = '#333333'
light_gray = '#F5F5F5'

# Title
ax.text(5, 9.5, 'Course Materials RAG System Architecture', 
        fontsize=20, fontweight='bold', ha='center', color=text_color)

# Document Ingestion Flow (Top)
ax.text(5, 8.8, 'Document Ingestion Flow', fontsize=16, fontweight='bold', 
        ha='center', color=secondary_color)

# Document box
doc_box = FancyBboxPatch((0.5, 7.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                         facecolor=light_gray, edgecolor=primary_color, linewidth=2)
ax.add_patch(doc_box)
ax.text(1.25, 8.1, 'Course\nDocuments\n(.txt, .pdf)', fontsize=10, ha='center', va='center')

# DocumentProcessor box
proc_box = FancyBboxPatch((2.5, 7.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                          facecolor=accent_color, edgecolor=primary_color, linewidth=2)
ax.add_patch(proc_box)
ax.text(3.25, 8.1, 'Document\nProcessor\n(Parse & Chunk)', fontsize=10, ha='center', va='center')

# VectorStore box
vector_box = FancyBboxPatch((4.5, 7.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                            facecolor=primary_color, edgecolor=primary_color, linewidth=2)
ax.add_patch(vector_box)
ax.text(5.25, 8.1, 'ChromaDB\nVector Store\n(Embeddings)', fontsize=10, ha='center', va='center', color='white')

# ChromaDB collections
catalog_box = FancyBboxPatch((6.5, 8.1), 1.2, 0.3, boxstyle="round,pad=0.05", 
                             facecolor=light_gray, edgecolor=primary_color, linewidth=1)
ax.add_patch(catalog_box)
ax.text(7.1, 8.25, 'course_catalog', fontsize=8, ha='center', va='center')

content_box = FancyBboxPatch((6.5, 7.7), 1.2, 0.3, boxstyle="round,pad=0.05", 
                             facecolor=light_gray, edgecolor=primary_color, linewidth=1)
ax.add_patch(content_box)
ax.text(7.1, 7.85, 'course_content', fontsize=8, ha='center', va='center')

# Arrows for ingestion flow
ax.arrow(2, 8.1, 0.4, 0, head_width=0.05, head_length=0.05, fc=text_color, ec=text_color)
ax.arrow(4, 8.1, 0.4, 0, head_width=0.05, head_length=0.05, fc=text_color, ec=text_color)
ax.arrow(6, 8.1, 0.4, 0, head_width=0.05, head_length=0.05, fc=text_color, ec=text_color)

# Query Processing Flow (Middle)
ax.text(5, 6.8, 'Query Processing Flow', fontsize=16, fontweight='bold', 
        ha='center', color=secondary_color)

# User query
user_box = FancyBboxPatch((0.5, 5.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                          facecolor=light_gray, edgecolor=secondary_color, linewidth=2)
ax.add_patch(user_box)
ax.text(1.25, 6.1, 'User Query\n"How does\nretrieval work?"', fontsize=10, ha='center', va='center')

# RAG System
rag_box = FancyBboxPatch((2.5, 5.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                         facecolor=secondary_color, edgecolor=secondary_color, linewidth=2)
ax.add_patch(rag_box)
ax.text(3.25, 6.1, 'RAG System\n(Orchestrator)', fontsize=10, ha='center', va='center', color='white')

# AI Generator with Tools
ai_box = FancyBboxPatch((4.5, 5.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                        facecolor=accent_color, edgecolor=accent_color, linewidth=2)
ax.add_patch(ai_box)
ax.text(5.25, 6.1, 'Claude AI\n+ Search Tools', fontsize=10, ha='center', va='center')

# Response
response_box = FancyBboxPatch((6.5, 5.8), 1.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=light_gray, edgecolor=primary_color, linewidth=2)
ax.add_patch(response_box)
ax.text(7.25, 6.1, 'AI Response\n+ Sources', fontsize=10, ha='center', va='center')

# Query flow arrows
ax.arrow(2, 6.1, 0.4, 0, head_width=0.05, head_length=0.05, fc=text_color, ec=text_color)
ax.arrow(4, 6.1, 0.4, 0, head_width=0.05, head_length=0.05, fc=text_color, ec=text_color)
ax.arrow(6, 6.1, 0.4, 0, head_width=0.05, head_length=0.05, fc=text_color, ec=text_color)

# Detailed Components (Bottom)
ax.text(5, 4.8, 'Core Components', fontsize=16, fontweight='bold', 
        ha='center', color=secondary_color)

# Component boxes
components = [
    (1, 3.8, 'FastAPI\nWeb Server\n(app.py)', primary_color),
    (3, 3.8, 'Session\nManager\n(History)', accent_color),
    (5, 3.8, 'Search Tools\n(Vector Search)', secondary_color),
    (7, 3.8, 'Config\n(Settings)', light_gray),
    (2, 2.8, 'Models\n(Data Classes)', light_gray),
    (4, 2.8, 'AI Generator\n(Claude API)', accent_color),
    (6, 2.8, 'Vector Store\n(ChromaDB)', primary_color),
]

for x, y, label, color in components:
    if color == light_gray:
        text_color_box = text_color
    else:
        text_color_box = 'white'
    
    comp_box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6, boxstyle="round,pad=0.05", 
                              facecolor=color, edgecolor=primary_color, linewidth=1)
    ax.add_patch(comp_box)
    ax.text(x, y, label, fontsize=9, ha='center', va='center', color=text_color_box)

# Data flow connections with curved arrows
# Vector store to search tools
connection1 = ConnectionPatch((5.25, 7.8), (5, 4.1), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5, 
                             connectionstyle="arc3,rad=0.3", color=primary_color, linewidth=2)
ax.add_artist(connection1)

# Search tools to AI
connection2 = ConnectionPatch((5, 3.5), (5.25, 5.8), "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5, 
                             connectionstyle="arc3,rad=-0.3", color=secondary_color, linewidth=2)
ax.add_artist(connection2)

# Add legend
legend_elements = [
    patches.Patch(color=primary_color, label='Core Storage & API'),
    patches.Patch(color=secondary_color, label='RAG Logic & Search'),
    patches.Patch(color=accent_color, label='AI & Processing'),
    patches.Patch(color=light_gray, label='Configuration & Models')
]
ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0))

# Add workflow annotations
ax.text(1.25, 7.5, '1. Parse', fontsize=8, ha='center', color=text_color, weight='bold')
ax.text(3.25, 7.5, '2. Chunk', fontsize=8, ha='center', color=text_color, weight='bold')
ax.text(5.25, 7.5, '3. Embed', fontsize=8, ha='center', color=text_color, weight='bold')
ax.text(7.1, 7.5, '4. Store', fontsize=8, ha='center', color=text_color, weight='bold')

ax.text(1.25, 5.5, '1. Query', fontsize=8, ha='center', color=text_color, weight='bold')
ax.text(3.25, 5.5, '2. Route', fontsize=8, ha='center', color=text_color, weight='bold')
ax.text(5.25, 5.5, '3. Search', fontsize=8, ha='center', color=text_color, weight='bold')
ax.text(7.25, 5.5, '4. Generate', fontsize=8, ha='center', color=text_color, weight='bold')

# Add document format example
ax.text(0.5, 1.5, 'Document Format:', fontsize=10, weight='bold', color=text_color)
ax.text(0.5, 1.2, 'Course Title: Advanced AI\nCourse Instructor: Dr. Smith\nLesson 1: Introduction\n  [lesson content...]\nLesson 2: Deep Learning\n  [lesson content...]', 
        fontsize=8, color=text_color, family='monospace', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor=light_gray, alpha=0.8))

# Add technical details
ax.text(8.5, 1.5, 'Technical Stack:', fontsize=10, weight='bold', color=text_color)
ax.text(8.5, 1.2, '• FastAPI (Web Framework)\n• ChromaDB (Vector Database)\n• Claude AI (LLM)\n• Sentence Transformers\n• Pydantic (Data Models)', 
        fontsize=8, color=text_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=light_gray, alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/kylekacius/Documents/dl-course-rag/backend/rag_system_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("System architecture diagram saved as 'rag_system_diagram.png'")