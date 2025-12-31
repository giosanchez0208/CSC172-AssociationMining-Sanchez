# CSC172 Association Rule Mining Project Progress Report
**Student:** Gio Kiefer A. Sanchez, 2022-0025
**Date:** December 31, 2025
**Repository:** https://github.com/giokiefer/CSC172_AssociationMining

## Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 39,774 transactions processed |
| Data Preprocessing | ✅ Completed | Cluster-based undersampling applied |
| EDA & Visualization | ✅ Completed | Flavor network and sparsity analyzed |
| Apriori Implementation | ✅ Completed | Canonical from-scratch implementation |
| Rule Evaluation | ✅ Completed | 1,735 rules generated and ranked |


## 1. Dataset Progress
- **Total transactions:** 39,774 (original), 9,340 (balanced for mining)
- **Unique items:** 6,714 unique ingredients (normalized via LLM and FoodOn), 20 cuisine types
- **Matrix size:** High-dimensional sparse matrix with >99% sparsity
- **Preprocessing applied:**
        - Hypergraph structure (node-data, edge-dict) converted to transactional format
        - LLM-assisted ingredient normalization to consolidate 3,200 lexical variants
        - Stratified cluster-based undersampling using k-means to address 16.8x class imbalance
        - Heterogeneous item transformation (adding 'cuisine:' prefix for Class Association Rules)

**Sample transaction preview:**
Transaction 1 (italian): {'olive oil', 'garlic', 'tomato', 'basil', 'parmesan', 'mozzarella', 'oregano', 'cuisine:italian'}
Transaction 2 (mexican): {'cumin', 'chili powder', 'garlic', 'onion', 'lime', 'cilantro', 'tomato', 'cuisine:mexican'}
Transaction 3 (indian): {'turmeric', 'garam masala', 'cumin', 'coriander', 'ginger', 'garlic', 'onion', 'cuisine:indian'}


## 2. EDA Progress

**Key Findings:**

• Top 5 ingredients (by prevalence):
  1. Salt (40.3%)
  2. Garlic (35.2%)
  3. Onion (33.1%)
  4. Olive oil (21.4%)
  5. Water (approx. 20%)

• Average basket size: 10.8 ingredients (median: 10)
• Distribution: Power law distribution where top 49 ingredients account for 50.3% of usage
• Complexity: Recipes range from 1 to 56 ingredients

**Current Metrics:**
• Total transactions analyzed: 9,340 (balanced subset)
• Frequent itemsets discovered: 1,735 rules generated after filtering
• Rule Type Distribution: 82% Ingredient-to-Ingredient, 15% Classification (Ingr -> Cuisine), 15% Signature (Cuisine -> Ingr)
• Computation time: Total pipeline runtime approx 225s on standard hardware

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| Severe Class Imbalance (16.8x) | ✅ Fixed | Implemented k-means stratified undersampling to preserve within-cuisine diversity |
| High-Dimensional Sparsity | ✅ Fixed | Utilized optimized one-hot encoding and min_support thresholds (1%) |
| Lexical Variation in Items | ✅ Fixed | LLM-assisted normalization consolidated ~3,200 variants into canonical forms |
| Rule Interpretation | ✅ Fixed | Categorized rules into Ingredient-to-Ingredient, Classification, and Signature rules |

## 4. Next Steps (Future Research and Extensions)
- **Algorithm Optimization:** Benchmark current Apriori performance against FP-Growth or Eclat to improve computational efficiency on larger datasets.
- **Weighted Rule Mining:** Incorporate ingredient quantities and proportions as weights to refine rule significance beyond binary presence.
- **Culinary Heritage Application:** Deploy the framework on a purpose-collected dataset of Mindanao Tri-People recipes (Maranao, Higaonon, and Christian) to document local "flavor DNA".
- **Temporal Analysis:** Investigate sequential pattern mining to capture the procedural order of ingredient additions and preparation methods.
- **Taxonomic Integration:** Implement hierarchical mining using ingredient ontologies (e.g., FoodOn) to analyze patterns at the sub-cuisine or regional level.