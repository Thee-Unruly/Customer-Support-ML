import pandas as pd
import ftfy

# Load dataset
df = pd.read_csv(
    "C:\\Users\\ibrahim.fadhili\\OneDrive - Agile Business Solutions\\Desktop\\Customer Support\\data\\raw\\Tickets.csv"
)

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = ftfy.fix_text(text)                 # Fix encoding issues
    text = text.replace("\n", " ")             # Replace newlines with space
    text = text.replace("\\n", " ")            # Remove literal \n in strings
    text = " ".join(text.split())              # Normalize spaces
    return text.strip()

# Apply cleaning
for col in ["subject", "body", "answer"]:
    df[col] = df[col].astype(str).apply(clean_text)

# Keep only English rows
df = df[df["language"] == "en"].reset_index(drop=True)

# Save processed dataset (no escape characters, plain text)
df.to_csv(
    "C:\\Users\\ibrahim.fadhili\\OneDrive - Agile Business Solutions\\Desktop\\Customer Support\\data\\processed\\processed_tickets.csv",
    index=False,
    quoting=1  # QUOTE_ALL ensures clean saving
)

print(f"✅ Preprocessing complete! Filtered dataset contains only English rows (no \\n) → saved as processed_tickets.csv")
