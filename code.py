from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Create the transaction data
data = [
    ['Potatoes', 'Tomatoes', 'Cucumber', 'Broccoli'],
    ['Potatoes', 'Tomatoes', 'Carrot', 'Cucumber', 'Pumpkin'],
    ['Carrot', 'Cucumber', 'Broccoli', 'Pumpkin'],
    ['Potatoes', 'Tomatoes', 'Cucumber', 'Pumpkin'],
    ['Carrot', 'Broccoli', 'Pumpkin'],
    ['Potatoes', 'Carrot', 'Broccoli', 'Pumpkin'],
    ['Potatoes', 'Tomatoes', 'Pumpkin'],
    ['Potatoes', 'Tomatoes', 'Cucumber', 'Pumpkin']
]
# Preprocess the data into a one-hot encoded DataFrame
all_items = set(item for transaction in data for item in transaction)
encoded_data = pd.DataFrame([{item: (item in transaction) for item in all_items} for transaction in data])

# Generate frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(encoded_data, min_support=4/8, use_colnames=True)

# Calculate the total number of itemsets
num_itemsets = len(frequent_itemsets)

# Generate association rules with confidence > 90%
rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="confidence", min_threshold=0.9)

print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules with Confidence > 90%:")
print(rules)
