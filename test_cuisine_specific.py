import pandas as pd
from collections import Counter

# Load the data
df = pd.read_csv('dataset/prepared_recipes_cleaned_balanced.csv')

print("=" * 80)
print("CUISINE-SPECIFIC INGREDIENT IDENTIFICATION & PREVALENCE")
print("=" * 80)

# Build comprehensive cuisine-ingredient matrix
print("\nStep 1: Building cuisine-ingredient frequency matrix...")
cuisine_ingredient_counts = {}

for cuisine in df['cuisine'].unique():
    cuisine_df = df[df['cuisine'] == cuisine]
    ingredient_counter = Counter()
    
    for ingredients_str in cuisine_df['ingredients']:
        ingredients = ingredients_str.split(',')
        ingredient_counter.update(ingredients)
    
    cuisine_ingredient_counts[cuisine] = ingredient_counter

print(f"Found {len(cuisine_ingredient_counts)} cuisines")

# Calculate ingredient specificity for each cuisine
print("Step 2: Calculating ingredient specificity scores...")
cuisine_specific_ingredients = {}

for cuisine in sorted(df['cuisine'].unique())[:5]:  # Just first 5 cuisines for testing
    cuisine_total = len(df[df['cuisine'] == cuisine])
    cuisine_ingredients = cuisine_ingredient_counts[cuisine]
    
    print(f"  Processing {cuisine} ({cuisine_total} recipes, {len(cuisine_ingredients)} unique ingredients)")
    
    # For each ingredient in this cuisine, calculate:
    # 1. Prevalence within cuisine (% of recipes)
    # 2. Specificity (how much more common in this cuisine vs others)
    ingredient_scores = []
    
    for ingredient, count in cuisine_ingredients.items():
        prevalence = (count / cuisine_total) * 100
        
        # Calculate average prevalence in other cuisines
        other_prevalence_sum = 0
        other_cuisines_count = 0
        
        for other_cuisine in df['cuisine'].unique():
            if other_cuisine != cuisine:
                other_total = len(df[df['cuisine'] == other_cuisine])
                other_count = cuisine_ingredient_counts[other_cuisine].get(ingredient, 0)
                other_prevalence = (other_count / other_total) * 100
                other_prevalence_sum += other_prevalence
                other_cuisines_count += 1
        
        avg_other_prevalence = other_prevalence_sum / other_cuisines_count if other_cuisines_count > 0 else 0
        
        # Specificity score: how much more common in this cuisine
        specificity = prevalence - avg_other_prevalence if avg_other_prevalence > 0 else prevalence
        
        # Only consider ingredients with reasonable prevalence and specificity
        if prevalence >= 20 and specificity >= 15:  # At least 20% in cuisine, 15% more than others
            ingredient_scores.append({
                'ingredient': ingredient,
                'prevalence': prevalence,
                'specificity': specificity,
                'count': count
            })
    
    # Sort by specificity score and keep top candidates
    ingredient_scores.sort(key=lambda x: x['specificity'], reverse=True)
    cuisine_specific_ingredients[cuisine] = ingredient_scores[:5]  # Top 5 per cuisine
    print(f"    Found {len(ingredient_scores)} candidate ingredients")

# Display results
print("\n" + "=" * 90)
print(f"{'Ingredient':<25} {'Cuisine':<15} {'Prevalence':<12} {'Specificity':<12} {'Count':<10}")
print("=" * 90)

for cuisine in sorted(cuisine_specific_ingredients.keys()):
    if cuisine_specific_ingredients[cuisine]:
        for ing_info in cuisine_specific_ingredients[cuisine]:
            print(f"{ing_info['ingredient']:<25} {cuisine:<15} {ing_info['prevalence']:>7.1f}%     "
                  f"{ing_info['specificity']:>7.1f}%     {ing_info['count']:<10}")
        print("-" * 90)

print("\n" + "=" * 90)
print("INTERPRETATION:")
print("- Prevalence: % of recipes in that cuisine containing the ingredient")
print("- Specificity: How much more common in this cuisine vs. average of others")
print("- Higher specificity = more characteristic of that cuisine")
print("=" * 90)
