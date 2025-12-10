
# CSC173 Deep Computer Vision Project Proposal
**Student:** Gio Kiefer A. Sanchez, 2022-0025
**Date:** 12/11/2025

## 1. Project Title 
Mining Culinary DNA: High-Dimensional Association Rule Framework for Flavor Profiling

## 2. Problem Statement
Urbanization and globalized supply chains risk homogenizing distinct regional food practices. There is no widely used quantitative pipeline to capture the structural rules that define a cuisine’s ingredient relationships, so cultural flavor signatures remain difficult to compare, preserve, or incorporate into recommender systems. This project will treat recipes as high-dimensional transactions and extract statistically significant association rules that reveal latent ingredient dependencies and flavor building blocks, providing a reproducible engine for computational culinary preservation and recommendation. The resulting framework is designed to be directly applicable to local Mindanao recipe archives after validation on the standard "What’s Cooking" dataset.

## 3. Objectives
- Construct a sparse, binary transaction matrix from raw recipe ingredient lists and implement robust ingredient normalization.  
- Implement and optimize Association Rule Mining (Apriori or equivalent) for high-dimensional, >99% sparse data to extract frequent itemsets and rules.  
- Quantify rule utility using Support, Confidence, Lift, and Conviction; prioritize rules that reflect culturally meaningful flavor signatures.  
- Visualize ingredient networks and produce a reproducible pipeline (code + notebook) for later application to local Mindanao recipe data.

## 4. Dataset Plan
- **Source:** Kaggle “What’s Cooking?” (Yummly recipes) and the curated hypergraph archive `cat-edge-Cooking.zip`. Dataset contains **39,774 recipes** (hyperedges) over **6,714 unique ingredients** (nodes).  
- **Classes:** 20 cuisine categories used as labels for cross-cuisine analysis and validation:
  - Italian  
  - Mexican  
  - Southern_US  
  - Indian  
  - Chinese  
  - French  
  - Cajun_Creole  
  - Thai  
  - Japanese  
  - Greek  
  - Spanish  
  - Korean  
  - Vietnamese  
  - Moroccan  
  - British  
  - Filipino  
  - Irish  
  - Jamaican  
  - Russian  
  - Brazilian  
- **Acquisition:**  
  - Original dataset from Kaggle (What’s Cooking? competition) — accessible at: https://www.kaggle.com/c/whats-cooking/data  
  - Archived hypergraph version available at Zenodo: https://zenodo.org/records/10157609  

## 5. Technical Approach

### Architecture & Pipeline

**Phase 1: Ingredient Normalization via Local LLM**
- Use local transformer (this project will use Deepseek) to standardize raw ingredient names in batches of 30 entries
- Design deterministic prompt template with few-shot examples (e.g., "fingerling potatoes" → "potato", "extra virgin olive oil" → "olive oil")
- Process all 6,714 unique ingredients incrementally; store results in new `standardized_ingredient` column alongside original names for data lineage
- Build expandable ingredient vocabulary mapping (original → standardized) saved as versioned CSV for reproducibility
- Validation: manually curate top 100 most-frequent raw ingredients as seed vocabulary; re-normalize random 5% sample after full processing to verify consistency

**Phase 2: Sparse Binary Transaction Matrix Construction**
- Input: 39,774 recipes with standardized ingredient lists (~10–12 ingredients per recipe on average)
- Use `mlxtend.preprocessing.TransactionEncoder` with `sparse=True` parameter to create one-hot binary encoding
- Output: Sparse DataFrame where rows = recipes, columns = ~400 standardized ingredient types
- Store underlying matrix in scipy CSR (Compressed Sparse Row) format to minimize memory footprint
  - Example memory footprint: 39,774 recipes × 400 ingredients = 15.9M cells; only ~120K non-zero entries (0.75% actual density)
  - CSR format reduces storage from ~2GB (dense) to ~20MB (sparse), enabling Colab execution within 12GB RAM budget
- Retain ingredient names as column headers for interpretability downstream
- **Before/After Statistics:**
  - Raw ingredients: 6,714 unique terms → Standardized: ~400 canonical types (94% reduction)
  - Sparsity: 99.25% sparse (only 120K non-zero entries in 15.9M cells)
  - Memory optimization: Dense matrix ~2GB → Sparse CSR ~20MB (100× reduction)
  - Recipes per cuisine: Italian 1,400, Indian 800, Mexican 600, ... (range: 200–1,400 recipes)

**Phase 2.5: Exploratory Data Analysis**
- Generate ingredient frequency distribution (bar chart: top-50 ingredients by occurrence count across all cuisines)
- Analyze recipe length distribution (histogram: number of ingredients per recipe, calculate mean/median/std)
- Compare cuisine sizes (bar chart: recipe count per 20 cuisines)
- Visualize sparsity pattern (heatmap: 100 random recipes × 100 most-frequent ingredients)
- Document key statistics:
  - Mean ingredients per recipe: ~10–12
  - Median ingredients per recipe
  - Standard deviation of recipe length
  - Global ingredient vocabulary reduction percentage
  - Sparsity percentage of final transaction matrix
- Generate visualizations: matplotlib/seaborn histograms, bar charts, and heatmaps for inclusion in final report

**Phase 3: Association Rule Mining (Frequent Itemset Extraction)**
- Primary Algorithm: **Apriori** via `mlxtend.frequent_patterns.apriori()`
  - Classic frequent itemset mining algorithm using candidate generation and pruning
  - Parameters: `min_support=0.005–0.02` (adaptive per cuisine based on recipe count), `max_len=3` (limit itemsets to 3-ingredient combinations for memory efficiency and actionability)
  - Output: frequent itemsets with support scores

- Performance Comparison: **FP-Growth** on same cuisines
  - Demonstrate performance improvements in execution time and memory usage vs. Apriori
  - Validate that both algorithms produce identical frequent itemsets
  - Document trade-offs and scalability characteristics
  - Compare on Italian and Indian cuisines as representative examples

- Rule Generation: Apply `mlxtend.association_rules()` with metrics:
  - **Support:** fraction of recipes containing itemset (minimum 0.5–2% depending on cuisine)
  - **Confidence:** P(consequent | antecedent), minimum 50–70%
  - **Lift:** confidence / P(consequent), prioritize lift > 2.0
  - **Conviction:** asymmetric rule strength measuring implication direction, prioritize > 1.5
  - **Leverage:** Support(A ∪ B) - Support(A) × Support(B), measures co-occurrence above baseline; range [-1, 1], positive = association above random chance

**Phase 4: Dual-Network Visualization**

*Network 1 – Main Links (Food Pairing Patterns):*
- Simplified ingredient co-occurrence graph showing frequent ingredient pairs across all recipes in a cuisine
- Nodes: top 50–100 most frequent ingredients per cuisine (by support)
- Edges: weighted by co-occurrence frequency (number of recipes containing both ingredients)
- Interactive HTML visualization via PyVis: force-directed layout, node size ∝ ingredient frequency, edge width ∝ strength, node color by ingredient category (protein/carb/fat/herb/spice)
- Insight: reveals dominant cooking patterns and foundational flavor combinations (e.g., Italian: tomato–basil–garlic)

*Network 2 – Differentiator (Cuisine Signatures):*
- Filter association rules by **cuisine-specific relative lift** to isolate unique ingredient relationships
- Calculation: for each rule, compute lift relative to cross-cuisine (global) baseline
  - Lift_cuisine = (P(A and B in cuisine X)) / (P(A) × P(B))
  - Lift_global = (P(A and B across all 39,774 recipes)) / (P(A) × P(B))
  - Retain only rules where Lift_cuisine / Lift_global > 3.0 (highly cuisine-specific)
- Nodes: ingredients with differentiator strength > 3.0 relative to baseline
- Edges: weighted by conviction (directional dependencies)
- Interactive sliders in PyVis to filter by lift threshold dynamically
- Insight: isolates cuisine-defining flavor signatures distinct from global patterns (e.g., Filipino: palapa + coconut + chilies as unique combination)

### Model & Algorithm Selection

| Component | Choice | Justification |
|-----------|--------|---------------|
| Ingredient Normalization | Local LLM (Deepseek/Llama 2) | Avoids token costs, full reproducibility, custom vocabulary control |
| Transaction Encoding | mlxtend TransactionEncoder (sparse=True) | Native sparse output, battle-tested, minimal dependencies |
| Frequent Itemset Mining | Apriori (primary), FP-Growth (comparison) | Apriori is foundational ARM algorithm; FP-Growth demonstrates optimization for sparse data |
| Rule Evaluation | Support/Confidence/Lift/Conviction/Leverage | Comprehensive ARM metrics; relative lift isolates cuisine signatures |
| Visualization | PyVis (interactive HTML) + NetworkX | Lightweight, interactive, suitable for Colab output |

### Framework & Dependencies

- **Core**: Python 3.8+, pandas, numpy, scipy (sparse matrix operations)
- **Association Rules**: mlxtend (frequent_patterns, association_rules)
- **Graph**: networkx (centrality analysis, community detection)
- **Visualization**: pyvis (interactive networks), matplotlib, seaborn (static plots)
- **LLM**: Local transformer (Deepseek, Llama 2, or similar; run locally or Colab)
- **Data I/O**: requests (Kaggle API), scikit-learn (optional, for validation classifier)

### Hardware & Compute Strategy

- **Primary**: Google Colab with T4 GPU
  - 12.7 GB RAM (effective ~11 GB usable)
  - T4 GPU: 16 GB VRAM for LLM inference
  - 68 GB storage for dataset + intermediate outputs

- **Memory Management**:
  - Load Kaggle dataset once; keep as sparse CSR matrices in memory
  - Process cuisines sequentially (1 at a time); save intermediate results (itemsets, rules) to CSV after each
  - Call `gc.collect()` after each cuisine to free memory
  - Monitor with `psutil.virtual_memory()` in loop; abort if > 10 GB threshold

- **LLM Inference**:
  - Run local LLM in Colab via `ollama` or HuggingFace transformers library
  - Process 30 ingredients per batch; log normalization mappings incrementally
  - Experiment with batch size (test 20, 30, 40) to optimize speed-memory trade-off

### Reproducible Pipeline (Jupyter Notebook)

1. **Setup**: Data loading via Kaggle API, dependency imports
2. **Normalization**: Batch LLM ingredient standardization with progress tracking; save mapping CSV
3. **Transaction Matrix**: Sparse one-hot encoding via TransactionEncoder → CSR format; verify memory footprint
4. **Exploratory Analysis**: Generate EDA visualizations (distributions, heatmaps); document statistics
5. **Association Rules**: Apriori algorithm per cuisine with adaptive min_support; export frequent itemsets
6. **Algorithm Comparison**: Run FP-Growth on select cuisines; compare execution time, memory, output equivalence
7. **Rule Filtering**: Compute relative lift and leverage; rank by cuisine-specific strength
8. **Visualization**: Generate dual networks (main links + differentiators) per cuisine; interactive PyVis outputs
9. **Export**: Top-20 rules per cuisine (CSV with all metrics including leverage), ingredient vocabulary, network data
10. **Validation (optional)**: Train cuisine classifier (Logistic Regression) using rules as features; report accuracy as rule quality proxy


## 6. Expected Challenges & Mitigations

| Challenge | Potential Issue | Mitigation Strategy |
|-----------|-----------------|-------------------|
| **LLM Normalization Inconsistency** | Deepseek may produce inconsistent standardizations across batches or for edge cases ("extra virgin" → "premium oil" vs. "virgin oil") | Build seed vocabulary from top 100 most-frequent raw ingredients; include 5–10 few-shot examples in LLM prompt to anchor behavior. After full processing, randomly sample 200 ingredients and manually verify. Re-normalize random 5% in second LLM pass to check consistency. Store all prompts/outputs in searchable CSV. |
| **Extreme Sparsity (>99%)** | With ~400 standardized ingredients and avg. ~10 per recipe, sparse transaction matrix produces very low support; Apriori may generate thousands of trivial rules or struggle to find any frequent itemsets | Use adaptive min_support per cuisine: `0.5% × (cuisine_recipe_count / 1000)` (e.g., Italian 1,400 recipes → 0.007). Pre-aggregate ingredient types (e.g., "onion variants" → "onion") to reduce dimensionality. Enforce `max_len=3` to limit itemsets. Post-mine, expand aggregates back to original names. Start with high support threshold; lower iteratively if < 20 frequent itemsets found. |
| **LLM Normalization Computation Time** | Processing 6,714 ingredients in batches of 30 via local LLM is slow; each batch may take 5–30 seconds depending on model size | Pre-process only top 1,000 most-frequent raw ingredients (covers ~90% of recipes); assign rare ingredients to standardized terms via edit-distance similarity matching. Use parallel LLM inference if GPU allows (multiprocessing or Colab's multi-GPU). Cache completed normalizations in pickle file; skip re-processing on notebook restart. Experiment with batch size (20, 30, 40) to optimize throughput. |
| **Sparse Matrix Memory Bottlenecks** | Converting all 39,774 recipes to dense one-hot before sparsification exhausts 12GB Colab RAM; failure to use CSR format wastes memory | Never construct dense intermediate matrix; use `TransactionEncoder(..., sparse=True)` for direct sparse output. Verify CSR memory footprint with `sparse_df.memory_usage(deep=True)` (target < 100 MB). Process one cuisine at a time (1–3K recipes per batch) → encode → mine → save results → clear memory. Monitor with `psutil.virtual_memory()` in loop; abort if threshold exceeds 10 GB. |
| **Identifying Culturally Meaningful Rules** | High-lift rules may be statistically significant but culinarily trivial ("salt occurs with everything"); important pairings may have low support due to recipe sparsity | Use relative lift filtering: compute lift per cuisine vs. global (cross-cuisine) baseline; retain only rules where cuisine_lift / global_lift > 3.0. Manually label ingredients by functional role (protein, starch, fat, acid, aromatic, umami); prefer rules spanning multiple roles. Validate against Filipino cuisine subset in Kaggle: extract rules and compare to known dishes (adobo ingredients: vinegar + soy sauce + bay leaf; sinigang: tamarind + tomato + fish) to confirm method captures domain patterns. |
| **Dual Visualization Interpretation** | Two separate networks (main links vs. differentiators) may confuse users; unclear how patterns relate across visualizations | Include clear caption and methodology section for each network explaining filtering criteria. Use synchronized node colors across both networks for ingredient tracking. Export all rules to single CSV with global *and* cuisine-specific metrics; users can pivot/filter independently. Provide heatmap of top ingredients per cuisine as alternative summary visualization. Document in notebook markdown cells. |
| **Transferability to Mindanao Data** | Mindanao recipes use region-specific ingredients (palapa, sakurab, tiyulah itum) not in Kaggle/Yummly dataset; ingredient vocabulary mismatch breaks pipeline | Design LLM normalization prompt to accept new ingredient categories; build standardized vocabulary as growing, user-extensible set (save as versioned CSV incrementally). Validate pipeline first on ~1,400 Filipino recipes in Kaggle; compare results to known Filipino dishes. Establish data-sharing agreement with Iligan City food vendors/culinary schools for Mindanao recipe crowdsourcing. Create "Mindanao ingredient appendix" documenting local ingredients and closest standardized equivalents (e.g., palapa ≈ "spice blend"). Include full LLM prompts, batch logs, vocabulary mappings in public GitHub repo for future researchers. |
---